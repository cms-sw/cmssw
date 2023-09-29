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
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

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
    const Setup* setup_ = nullptr;
    // class to encode layer ids used between DTC and TFP in Hybrid
    const LayerEncoding* layerEncoding_ = nullptr;
    // ED input token of TTStubs
    EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<TTDTC> edPutTokenAccepted_;
    // ED output token for lost stubs
    EDPutTokenT<TTDTC> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // configuration
    ParameterSet iConfig_;
  };

  ProducerED::ProducerED(const ParameterSet& iConfig) : iConfig_(iConfig) {
    // book in- and output ED products
    const auto& inputTag = iConfig.getParameter<InputTag>("InputTag");
    const auto& branchAccepted = iConfig.getParameter<string>("BranchAccepted");
    const auto& branchLost = iConfig.getParameter<string>("BranchLost");
    edGetToken_ = consumes<TTStubDetSetVec>(inputTag);
    edPutTokenAccepted_ = produces<TTDTC>(branchAccepted);
    edPutTokenLost_ = produces<TTDTC>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
  }

  void ProducerED::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
  }

  void ProducerED::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DTC products
    TTDTC productAccepted = setup_->ttDTC();
    TTDTC productLost = setup_->ttDTC();
    if (setup_->configurationSupported()) {
      // read in stub collection
      Handle<TTStubDetSetVec> handle;
      iEvent.getByToken(edGetToken_, handle);
      // apply cabling map, reorganise stub collections
      vector<vector<vector<TTStubRef>>> stubsDTCs(setup_->numDTCs(),
                                                  vector<vector<TTStubRef>>(setup_->numModulesPerDTC()));
      for (auto module = handle->begin(); module != handle->end(); module++) {
        // DetSetVec->detId + 1 = tk layout det id
        const DetId detId = module->detId() + setup_->offsetDetIdDSV();
        // corresponding sensor module
        SensorModule* sm = setup_->sensorModule(detId);
        // empty stub collection
        vector<TTStubRef>& stubsModule = stubsDTCs[sm->dtcId()][sm->modId()];
        stubsModule.reserve(module->size());
        for (TTStubDetSet::const_iterator ttStub = module->begin(); ttStub != module->end(); ttStub++)
          stubsModule.emplace_back(makeRefTo(handle, ttStub));
      }
      // board level processing
      for (int dtcId = 0; dtcId < setup_->numDTCs(); dtcId++) {
        // create single outer tracker DTC board
        DTC dtc(iConfig_, setup_, layerEncoding_, dtcId, stubsDTCs.at(dtcId));
        // route stubs and fill products
        dtc.produce(productAccepted, productLost);
      }
    }
    // store ED products
    iEvent.emplace(edPutTokenAccepted_, std::move(productAccepted));
    iEvent.emplace(edPutTokenLost_, std::move(productLost));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::ProducerED);
