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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerDTC
   *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class ProducerDTC : public stream::EDProducer<> {
  public:
    explicit ProducerDTC(const ParameterSet&);
    ~ProducerDTC() override {}

  private:
    void produce(Event&, const EventSetup&) override;
    // ED input token of TTStubs
    EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<TTDTC> edPutTokenAccepted_;
    // ED output token for lost stubs
    EDPutTokenT<TTDTC> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, SetupRcd> esGetTokenLayerEncoding_;
  };

  ProducerDTC::ProducerDTC(const ParameterSet& iConfig) {
    // book in- and output ED products
    const auto& inputTag = iConfig.getParameter<InputTag>("InputTag");
    const auto& branchAccepted = iConfig.getParameter<string>("BranchAccepted");
    const auto& branchLost = iConfig.getParameter<string>("BranchLost");
    edGetToken_ = consumes<TTStubDetSetVec>(inputTag);
    edPutTokenAccepted_ = produces<TTDTC>(branchAccepted);
    edPutTokenLost_ = produces<TTDTC>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerDTC::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // class to encode layer ids used between DTC and TFP in Hybrid
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty DTC products
    TTDTC productAccepted = setup->ttDTC();
    TTDTC productLost = setup->ttDTC();
    // read in stub collection
    Handle<TTStubDetSetVec> handle;
    iEvent.getByToken(edGetToken_, handle);
    // apply cabling map, reorganise stub collections
    vector<vector<vector<TTStubRef>>> stubsDTCs(setup->numDTCs(), vector<vector<TTStubRef>>(setup->numModulesPerDTC()));
    for (auto module = handle->begin(); module != handle->end(); module++) {
      // DetSetVec->detId + 1 = tk layout det id
      const DetId detId = module->detId() + setup->offsetDetIdDSV();
      // corresponding sensor module
      SensorModule* sm = setup->sensorModule(detId);
      // empty stub collection
      vector<TTStubRef>& stubsModule = stubsDTCs[sm->dtcId()][sm->modId()];
      stubsModule.reserve(module->size());
      for (TTStubDetSet::const_iterator ttStub = module->begin(); ttStub != module->end(); ttStub++)
        stubsModule.emplace_back(makeRefTo(handle, ttStub));
    }
    // board level processing
    for (int dtcId = 0; dtcId < setup->numDTCs(); dtcId++) {
      // create single outer tracker DTC board
      DTC dtc(setup, dataFormats, layerEncoding, dtcId, stubsDTCs.at(dtcId));
      // route stubs and fill products
      dtc.produce(productAccepted, productLost);
    }
    // store ED products
    iEvent.emplace(edPutTokenAccepted_, std::move(productAccepted));
    iEvent.emplace(edPutTokenLost_, std::move(productLost));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::ProducerDTC);
