#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

using namespace std;
using namespace edm;

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerED
   *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class ProducerED : public stream::EDProducer<> {
  public:
    explicit ProducerED(const ParameterSet&);
    ~ProducerED() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}
    // helper class to store configurations
    Setup setup_;
    // ED input token of TTStubs
    EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<TTDTC> edPutTokenAccepted_;
    // ED output token for lost stubs
    EDPutTokenT<TTDTC> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetToken_;
    // configuration
    ParameterSet iConfig_;
    // throws an exception if current configuration inconsitent with history
    bool checkHistory_;
  };

  ProducerED::ProducerED(const ParameterSet& iConfig)
      : iConfig_(iConfig), checkHistory_(iConfig.getParameter<bool>("CheckHistory")) {
    // book in- and output ED products
    const auto& inputTag = iConfig.getParameter<InputTag>("InputTag");
    const auto& branchAccepted = iConfig.getParameter<string>("BranchAccepted");
    const auto& branchLost = iConfig.getParameter<string>("BranchLost");
    edGetToken_ = consumes<TTStubDetSetVec>(inputTag);
    edPutTokenAccepted_ = produces<TTDTC>(branchAccepted);
    edPutTokenLost_ = produces<TTDTC>(branchLost);
    // book ES product
    esGetToken_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
  }

  void ProducerED::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = iSetup.getData(esGetToken_);
    if (!setup_.configurationSupported())
      return;
    // check process history if desired
    if (checkHistory_)
      setup_.checkHistory(iRun.processHistory());
  }

  void ProducerED::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DTC products
    TTDTC productAccepted = setup_.ttDTC();
    TTDTC productLost = setup_.ttDTC();
    if (setup_.configurationSupported()) {
      // read in stub collection
      Handle<TTStubDetSetVec> handle;
      iEvent.getByToken(edGetToken_, handle);
      // apply cabling map, reorganise stub collections
      vector<vector<vector<TTStubRef>>> stubsDTCs(setup_.numDTCs(),
                                                  vector<vector<TTStubRef>>(setup_.numModulesPerDTC()));
      for (auto module = handle->begin(); module != handle->end(); module++) {
        // DetSetVec->detId + 1 = tk layout det id
        const DetId detId = module->detId() + setup_.offsetDetIdDSV();
        // corresponding sensor module
        SensorModule* sm = setup_.sensorModule(detId);
        // empty stub collection
        vector<TTStubRef>& stubsModule = stubsDTCs[sm->dtcId()][sm->modId()];
        stubsModule.reserve(module->size());
        for (TTStubDetSet::const_iterator ttStub = module->begin(); ttStub != module->end(); ttStub++)
          stubsModule.emplace_back(makeRefTo(handle, ttStub));
      }
      // board level processing
      for (int dtcId = 0; dtcId < setup_.numDTCs(); dtcId++) {
        // create single outer tracker DTC board
        DTC dtc(iConfig_, setup_, dtcId, stubsDTCs.at(dtcId));
        // route stubs and fill products
        dtc.produce(productAccepted, productLost);
      }
    }
    // store ED products
    iEvent.emplace(edPutTokenAccepted_, move(productAccepted));
    iEvent.emplace(edPutTokenLost_, move(productLost));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::ProducerED);