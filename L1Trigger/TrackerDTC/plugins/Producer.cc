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
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <memory>
#include <numeric>
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
    // collection of outer tracker sensor modules
    std::vector<Module> modules_;
    // ED in- and output tokens
    EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    EDPutTokenT<TTDTC> putTokenTTDTC_;
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
    putTokenTTDTC_ = produces<TTDTC>(settings_.productBranch());
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
    // convert data fromat specific stuff
    settings_.beginRun();
    // check coniguration
    settings_.checkConfiguration();
    if (!settings_.configurationSupported())
      return;
    // convert cabling map
    settings_.convertCablingMap();
    // convert outer tracker geometry
    const vector<DetId>& cablingMap = settings_.cablingMap();
    auto acc = [](int& sum, const DetId& detId) { return sum += !detId.null(); };
    const int numModules = accumulate(cablingMap.begin(), cablingMap.end(), 0, acc);
    modules_.reserve(numModules);
    int modId(0);
    for (const DetId& detId : cablingMap) {
      if (!detId.null())
        modules_.emplace_back(&settings_, detId, modId);
      modId++;
    }
    settings_.setModules(modules_);
  }

  void Producer::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DTC product
    TTDTC product(settings_.numRegions(), settings_.numOverlappingRegions(), settings_.numDTCsPerRegion());

    if (settings_.configurationSupported()) {
      // read in stub collection
      Handle<TTStubDetSetVec> handleTTStubDetSetVec;
      iEvent.getByToken(getTokenTTStubDetSetVec_, handleTTStubDetSetVec);

      // apply cabling map
      auto acc = [](int& sum, const TTStubDetSet* module) { return sum += module ? module->size() : 0; };
      vector<vector<const TTStubDetSet*>> ttDTCs(settings_.numDTCs(),
                                                 vector<const TTStubDetSet*>(settings_.numModulesPerDTC(), nullptr));
      TTStubDetSetVec::const_iterator ttModule;
      for (ttModule = handleTTStubDetSetVec->begin(); ttModule != handleTTStubDetSetVec->end(); ttModule++) {
        // DetSetVec->detId + 1 = tk layout det id
        const DetId detId = ttModule->detId() + settings_.offsetDetIdDSV();
        // outer tracker module id [0-15551]
        const int modId = settings_.modId(detId);
        // outer tracker dtc id [0-215]
        const int dtcId = modId / settings_.numModulesPerDTC();
        // outer tracker dtc channel id [0-71]
        const int channelId = modId % settings_.numModulesPerDTC();
        ttDTCs[dtcId][channelId] = &*ttModule;
      }

      // read in and convert event content
      int dtcId(0);
      for (const vector<const TTStubDetSet*>& ttDTC : ttDTCs) {
        // get modules connected to this dtc
        const vector<Module*> modules = settings_.modules(dtcId);
        // count number of stubs on this dtc
        const int nSubs = accumulate(ttDTC.begin(), ttDTC.end(), 0, acc);
        // create single outer tracker DTC board
        DTC dtc(&settings_, nSubs);
        // fill incoming stubs over all channel
        int channelId(0);
        for (const TTStubDetSet* ttModule : ttDTC) {
          // create TTStubRefs from one module
          vector<TTStubRef> ttStubRefs;
          if (ttModule) {
            ttStubRefs.reserve(ttModule->size());
            for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++)
              ttStubRefs.emplace_back(makeRefTo(handleTTStubDetSetVec, ttStub));
            // truncate incoming stubs if desired
            if (settings_.enableTruncation())
              ttStubRefs.resize(std::min((int)ttStubRefs.size(), settings_.maxFramesChannelInput()));
          }
          // fill incoming stubs of this channel
          dtc.consume(ttStubRefs, modules.at(channelId++));
        }
        // route stubs and fill product
        dtc.produce(product, dtcId++);
      }
    }

    // store ED product
    iEvent.emplace(putTokenTTDTC_, move(product));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Producer);