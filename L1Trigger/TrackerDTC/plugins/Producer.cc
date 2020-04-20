#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;
using namespace edm;

namespace trackerDTC {

  /*! \class  trackerDTC::Producer
   *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class Producer : public stream::EDProducer<> {
  public:
    explicit Producer(const ParameterSet&);
    ~Producer() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}
    // helper class to store configurations
    Settings settings_;
    // ED in- and output tokens
    EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    EDPutTokenT<TTDTC> putTokenTTDTCAccepted_;
    EDPutTokenT<TTDTC> putTokenTTDTCLost_;
    // ES tokens
    ESGetToken<TTStubAlgorithm<Ref_Phase2TrackerDigi_>, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
    ESGetToken<MagneticField, IdealMagneticFieldRecord> getTokenMagneticField_;
    ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    ESGetToken<DDCompactView, IdealGeometryRecord> getTokenGeometryConfiguration_;
  };

  Producer::Producer(const ParameterSet& iConfig) : settings_(iConfig) {
    // book in- and output ED products
    getTokenTTStubDetSetVec_ = consumes<TTStubDetSetVec>(settings_.inputTagTTStubDetSetVec());
    putTokenTTDTCAccepted_ = produces<TTDTC>(settings_.productBranchAccepted());
    putTokenTTDTCLost_ = produces<TTDTC>(settings_.productBranchLost());
    // book ES products
    getTokenTTStubAlgorithm_ =
        esConsumes<TTStubAlgorithm<Ref_Phase2TrackerDigi_>, TTStubAlgorithmRecord, Transition::BeginRun>(
            settings_.inputTagTTStubAlgorithm());
    getTokenMagneticField_ =
        esConsumes<MagneticField, IdealMagneticFieldRecord, Transition::BeginRun>(settings_.inputTagMagneticField());
    getTokenTrackerGeometry_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, Transition::BeginRun>(
        settings_.inputTagTrackerGeometry());
    getTokenTrackerTopology_ =
        esConsumes<TrackerTopology, TrackerTopologyRcd, Transition::BeginRun>(settings_.inputTagTrackerTopology());
    getTokenCablingMap_ =
        esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, Transition::BeginRun>(
            settings_.inputTagCablingMap());
    getTokenGeometryConfiguration_ =
        esConsumes<DDCompactView, IdealGeometryRecord, Transition::BeginRun>(settings_.inputTagGeometryConfiguration());
  }

  void Producer::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // read in detector parameter
    settings_.setTrackerGeometry(&iSetup.getData(getTokenTrackerGeometry_));
    settings_.setTrackerTopology(&iSetup.getData(getTokenTrackerTopology_));
    settings_.setMagneticField(&iSetup.getData(getTokenMagneticField_));
    settings_.setCablingMap(&iSetup.getData(getTokenCablingMap_));
    settings_.setTTStubAlgorithm(iSetup.getHandle(getTokenTTStubAlgorithm_));
    settings_.setGeometryConfiguration(iSetup.getHandle(getTokenGeometryConfiguration_));
    settings_.setProcessHistory(iRun.processHistory());
    // check coniguration
    settings_.checkConfiguration();
    if (!settings_.configurationSupported())
      return;
    // convert ES Products into handy objects
    settings_.beginRun();
  }

  void Producer::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DTC products
    TTDTC productAccepted(settings_.numRegions(), settings_.numOverlappingRegions(), settings_.numDTCsPerRegion());
    TTDTC productLost(settings_.numRegions(), settings_.numOverlappingRegions(), settings_.numDTCsPerRegion());
    if (settings_.configurationSupported()) {
      // read in stub collection
      Handle<TTStubDetSetVec> handleTTStubDetSetVec;
      iEvent.getByToken(getTokenTTStubDetSetVec_, handleTTStubDetSetVec);
      // apply cabling map
      vector<vector<vector<TTStubRef>>> ttDTCs(settings_.numDTCs(),
                                               vector<vector<TTStubRef>>(settings_.numModulesPerDTC()));
      for (TTStubDetSetVec::const_iterator module = handleTTStubDetSetVec->begin();
           module != handleTTStubDetSetVec->end();
           module++) {
        // DetSetVec->detId + 1 = tk layout det id
        const DetId detId = module->detId() + settings_.offsetDetIdDSV();
        // outer tracker module id [0-15551]
        int modId = settings_.modId(detId);
        // outer tracker dtc id [0-215]
        const int dtcId = modId / settings_.numModulesPerDTC();
        // outer tracker dtc channel id [0-71]
        const int channelId = modId % settings_.numModulesPerDTC();
        vector<TTStubRef>& ttModule = ttDTCs[dtcId][channelId];
        ttModule.reserve(module->size());
        for (TTStubDetSet::const_iterator ttStub = module->begin(); ttStub != module->end(); ttStub++)
          ttModule.emplace_back(makeRefTo(handleTTStubDetSetVec, ttStub));
      }
      // board level processing
      for (int dtcId = 0; dtcId < settings_.numDTCs(); dtcId++) {
        // create single outer tracker DTC board
        DTC dtc(&settings_, dtcId, ttDTCs[dtcId]);
        // route stubs and fill products
        dtc.produce(productAccepted, productLost);
      }
    }
    // store ED products
    iEvent.emplace(putTokenTTDTCAccepted_, move(productAccepted));
    iEvent.emplace(putTokenTTDTCLost_, move(productLost));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Producer);