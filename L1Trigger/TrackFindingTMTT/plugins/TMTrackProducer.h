#ifndef L1Trigger_TrackFindingTMTT_TMTrackProducer_h
#define L1Trigger_TrackFindingTMTT_TMTrackProducer_h

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Histos.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/GlobalCacheTMTT.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <vector>
#include <list>
#include <string>
#include <memory>

namespace tmtt {

  class TrackFitGeneric;

  class TMTrackProducer : public edm::stream::EDProducer<edm::GlobalCache<GlobalCacheTMTT>> {
  public:
    explicit TMTrackProducer(const edm::ParameterSet &, GlobalCacheTMTT const *globalCacheTMTT);
    ~TMTrackProducer() {}

    static std::unique_ptr<GlobalCacheTMTT> initializeGlobalCache(edm::ParameterSet const &iConfig);

    static void globalEndJob(GlobalCacheTMTT *globalCacheTMTT);

  private:
    typedef std::vector<TTTrack<Ref_Phase2TrackerDigi_>> TTTrackCollection;

    virtual void beginRun(const edm::Run &, const edm::EventSetup &);

    virtual void produce(edm::Event &, const edm::EventSetup &);

  private:
    bool debug_;

    // ES tokens
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
    // ED tokens
    edm::EDGetTokenT<TTStubDetSetVec> stubToken_;
    edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
    edm::EDGetTokenT<TTStubAssMap> stubTruthToken_;
    edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken_;
    edm::EDGetTokenT<reco::GenJetCollection> genJetToken_;

    // Info about tracker geometry
    const TrackerGeometry *trackerGeometry_;
    const TrackerTopology *trackerTopology_;
    std::list<TrackerModule> listTrackerModule_;

    // Configuration parameters
    Settings settings_;
    std::vector<std::string> trackFitters_;
    std::vector<std::string> useRZfilter_;
    bool runRZfilter_;

    Histos &hists_;
    HTrphi::ErrorMonitor &htRphiErrMon_;
    StubWindowSuggest &stubWindowSuggest_;

    std::map<std::string, std::unique_ptr<TrackFitGeneric>> fitterWorkerMap_;
  };

}  // namespace tmtt

#endif
