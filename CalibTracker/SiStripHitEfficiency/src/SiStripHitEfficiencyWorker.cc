////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitEfficiency
// Class:            HitEff (CalibTree production
// Original Author:  Keith Ulmer--University of Colorado
//                   keith.ulmer@colorado.edu
// Class:            SiStripHitEffFromCalibTree
// Original Author:  Christopher Edelmaier
//
///////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class SiStripHitEfficiencyWorker : public DQMEDAnalyzer {
public:
  explicit SiStripHitEfficiencyWorker(const edm::ParameterSet& conf);
  ~SiStripHitEfficiencyWorker() override;

private:
  void beginJob();  // TODO remove
  void endJob();    // TODO remove
  void bookHistograms(DQMStore::IBooker& booker, const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void fillForTraj(const TrajectoryAtInvalidHit& tm,
                   const TrackerTopology* tTopo,
                   const TrackerGeometry* tkgeom,
                   const StripClusterParameterEstimator* stripCPE,
                   const SiStripQuality* stripQuality,
                   const DetIdCollection& fedErrorIds,
                   const edm::Handle<edm::DetSetVector<SiStripRawDigi>>& commonModeDigis,
                   const edmNew::DetSetVector<SiStripCluster>& theClusters,
                   int bunchCrossing,
                   float instLumi,
                   float PU,
                   bool highPurity);

  TString layerName(unsigned int k) const;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>> commonModeToken_;

  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_;

  unsigned int _clusterMatchingMethod;
  float _ResXSig;
  float _clusterTrajDist;
  float _stripsApvEdge;
  bool _useOnlyHighPurityTracks;
  int _bunchx;
  bool _showRings;
  bool _showTOB6TEC9;

  std::set<uint32_t> badModules_;

  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory>> trajectories_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAsso_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digis_token_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> trackerEvent_token_;

  int events, EventTrackCKF;

  unsigned int layers;
  bool DEBUG;

  struct EffME1 {
    EffME1() : hTotal(nullptr), hFound(nullptr) {}
    EffME1(MonitorElement* total, MonitorElement* found) : hTotal(total), hFound(found) {}

    void fill(double x, bool found, float weight=1.) {
      hTotal->Fill(x, weight);
      if (found) {
        hFound->Fill(x, weight);
      }
    }

    MonitorElement *hTotal, *hFound;
  };
  struct EffTkMap {
    EffTkMap() : hTotal(nullptr), hFound(nullptr) {}
    EffTkMap(std::unique_ptr<TkHistoMap>&& total, std::unique_ptr<TkHistoMap>&& found) : hTotal(std::move(total)), hFound(std::move(found)) {}

    void fill(uint32_t id, bool found, float weight=1.) {
      hTotal->fill(id, weight);
      if (found) {
        hFound->fill(id, weight);
      }
    }

    std::unique_ptr<TkHistoMap> hTotal, hFound;
  };

  MonitorElement *h_bx, *h_instLumi, *h_PU;
  EffME1 h_goodLayer;
  EffME1 h_allLayer;
  EffME1 h_layer;
  std::vector<MonitorElement*> h_resolution;
  std::vector<EffME1> h_layer_vsLumi;
  std::vector<EffME1> h_layer_vsBx;
  std::vector<EffME1> h_layer_vsPU;
  std::vector<EffME1> h_layer_vsCM;
  std::vector<MonitorElement*> h_hotcold;

  EffTkMap h_module;
};

//
// constructors and destructor
//

SiStripHitEfficiencyWorker::SiStripHitEfficiencyWorker(const edm::ParameterSet& conf)
    : scalerToken_(consumes<LumiScalersCollection>(conf.getParameter<edm::InputTag>("lumiScalers"))),
      commonModeToken_(mayConsume<edm::DetSetVector<SiStripRawDigi>>(conf.getParameter<edm::InputTag>("commonMode"))),
      combinatorialTracks_token_(
          consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("combinatorialTracks"))),
      trajectories_token_(consumes<std::vector<Trajectory>>(conf.getParameter<edm::InputTag>("trajectories"))),
      trajTrackAsso_token_(consumes<TrajTrackAssociationCollection>(conf.getParameter<edm::InputTag>("trajectories"))),
      clusters_token_(
          consumes<edmNew::DetSetVector<SiStripCluster>>(conf.getParameter<edm::InputTag>("siStripClusters"))),
      digis_token_(consumes<DetIdCollection>(conf.getParameter<edm::InputTag>("siStripDigis"))),
      trackerEvent_token_(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))) {
  layers = conf.getParameter<int>("Layer");
  DEBUG = conf.getParameter<bool>("Debug");
  addLumi_ = conf.getUntrackedParameter<bool>("addLumi", false);
  addCommonMode_ = conf.getUntrackedParameter<bool>("addCommonMode", false);
  cutOnTracks_ = conf.getUntrackedParameter<bool>("cutOnTracks", false);
  trackMultiplicityCut_ = conf.getUntrackedParameter<unsigned int>("trackMultiplicity", 100);
  useFirstMeas_ = conf.getUntrackedParameter<bool>("useFirstMeas", false);
  useLastMeas_ = conf.getUntrackedParameter<bool>("useLastMeas", false);
  useAllHitsFromTracksWithMissingHits_ =
      conf.getUntrackedParameter<bool>("useAllHitsFromTracksWithMissingHits", false);

  // TODO make consistent

  const std::string badModulesFile = conf.getUntrackedParameter<std::string>("BadModulesFile", "");
  _clusterMatchingMethod = conf.getUntrackedParameter<int>("ClusterMatchingMethod", 0);
  _ResXSig = conf.getUntrackedParameter<double>("ResXSig", -1);
  _clusterTrajDist = conf.getUntrackedParameter<double>("ClusterTrajDist", 64.0);
  _stripsApvEdge = conf.getUntrackedParameter<double>("StripsApvEdge", 10.0);
  _useOnlyHighPurityTracks = conf.getUntrackedParameter<bool>("UseOnlyHighPurityTracks", true);
  _bunchx = conf.getUntrackedParameter<int>("BunchCrossing", 0);
  _showRings = conf.getUntrackedParameter<bool>("ShowRings", false);
  _showTOB6TEC9 = conf.getUntrackedParameter<bool>("ShowTOB6TEC9", false);

  if (!badModulesFile.empty()) {
    std::ifstream badModules_file(badModulesFile);
    uint32_t badmodule_detid;
    int mods, fiber1, fiber2, fiber3;
    if (badModules_file.is_open()) {
      std::string line;
      while (getline(badModules_file, line)) {
        if (badModules_file.eof())
          continue;
        std::stringstream ss(line);
        ss >> badmodule_detid >> mods >> fiber1 >> fiber2 >> fiber3;
        if (badmodule_detid != 0 && mods == 1 && (fiber1 == 1 || fiber2 == 1 || fiber3 == 1))
          badModules_.insert(badmodule_detid);
      }
      badModules_file.close();
    }
  }
  if (!badModules_.empty())
    std::cout << "Remove additionnal bad modules from the analysis: " << std::endl;
  for ( const auto badMod : badModules_ ) {
    std::cout << " " << badMod << std::endl;
  }
}

// Virtual destructor needed.
SiStripHitEfficiencyWorker::~SiStripHitEfficiencyWorker() {}

void SiStripHitEfficiencyWorker::beginJob() {
  // TODO convert to counters, or simply remove?
  events = 0;
  EventTrackCKF = 0;
}

namespace {

  double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr) {
    double error = sqrt(parameters.second.xx() + xerr * xerr);
    double separation = abs(parameters.first.x() - xx);
    double consistency = separation / error;
    return consistency;
  }

  bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) {
    unsigned int layer;
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        layer = tTopo->tibLayer(iidd);
        return (layer == 1 || layer == 2);
      case SiStripSubdetector::TOB:
        layer = tTopo->tobLayer(iidd) + 4;
        return (layer == 5 || layer == 6);
      case SiStripSubdetector::TID:
        layer = tTopo->tidRing(iidd) + 10;
        return (layer == 11 || layer == 12);
      case SiStripSubdetector::TEC:
        layer = tTopo->tecRing(iidd) + 13;
        return (layer == 14 || layer == 15 || layer == 18);
      default:
        return false;
    }
  }

  bool check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj) {
    unsigned int partner_iidd = 0;
    bool found2DPartner = false;
    // first get the id of the other detector
    if ((iidd & 0x3) == 1)
      partner_iidd = iidd + 1;
    if ((iidd & 0x3) == 2)
      partner_iidd = iidd - 1;
    // next look in the trajectory measurements for a measurement from that detector
    // loop through trajectory measurements to find the partner_iidd
    for (const auto& tm : traj) {
      if (tm.recHit()->geographicalId().rawId() == partner_iidd) {
        found2DPartner = true;
      }
    }
    return found2DPartner;
  }

  unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo) {
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        return tTopo->tibLayer(iidd);
      case SiStripSubdetector::TOB:
        return tTopo->tobLayer(iidd) + 4;
      case SiStripSubdetector::TID:
        return tTopo->tidWheel(iidd) + 10;
      case SiStripSubdetector::TEC:
        return tTopo->tecWheel(iidd) + 13;
      default:
        return 0;
    }
  }

  bool isInBondingExclusionZone(
      unsigned int iidd, unsigned int TKlayers, double yloc, double yErr, const TrackerTopology* tTopo) {
    constexpr float exclusionWidth = 0.4;
    constexpr float TOBexclusion = 0.0;
    constexpr float TECexRing5 = -0.89;
    constexpr float TECexRing6 = -0.56;
    constexpr float TECexRing7 = 0.60;

    //Added by Chris Edelmaier to do TEC bonding exclusion
    const int subdetector = ((iidd >> 25) & 0x7);
    const int ringnumber = ((iidd >> 5) & 0x7);

    bool inZone = false;
    //New TOB and TEC bonding region exclusion zone
    if ((TKlayers >= 5 && TKlayers < 11) || ((subdetector == 6) && ((ringnumber >= 5) && (ringnumber <= 7)))) {
      //There are only 2 cases that we need to exclude for
      float highzone = 0.0;
      float lowzone = 0.0;
      float higherr = yloc + 5.0 * yErr;
      float lowerr = yloc - 5.0 * yErr;
      if (TKlayers >= 5 && TKlayers < 11) {
        //TOB zone
        highzone = TOBexclusion + exclusionWidth;
        lowzone = TOBexclusion - exclusionWidth;
      } else if (ringnumber == 5) {
        //TEC ring 5
        highzone = TECexRing5 + exclusionWidth;
        lowzone = TECexRing5 - exclusionWidth;
      } else if (ringnumber == 6) {
        //TEC ring 6
        highzone = TECexRing6 + exclusionWidth;
        lowzone = TECexRing6 - exclusionWidth;
      } else if (ringnumber == 7) {
        //TEC ring 7
        highzone = TECexRing7 + exclusionWidth;
        lowzone = TECexRing7 - exclusionWidth;
      }
      //Now that we have our exclusion region, we just have to properly identify it
      if ((highzone <= higherr) && (highzone >= lowerr))
        inZone = true;
      if ((lowzone >= lowerr) && (lowzone <= higherr))
        inZone = true;
      if ((higherr <= highzone) && (higherr >= lowzone))
        inZone = true;
      if ((lowerr >= lowzone) && (lowerr <= highzone))
        inZone = true;
    }
    return inZone;
  }

  struct ClusterInfo {
    float xResidual;
    float xResidualPull;
    float xLocal;
    ClusterInfo(float xRes, float xResPull, float xLoc) : xResidual(xRes), xResidualPull(xResPull), xLocal(xLoc) {}
  };

  float calcPhi(float x, float y) {
    float phi = 0;
    float Pi = 3.14159;
    if ((x >= 0) && (y >= 0))
      phi = std::atan(y / x);
    else if ((x >= 0) && (y <= 0))
      phi = std::atan(y / x) + 2 * Pi;
    else if ((x <= 0) && (y >= 0))
      phi = std::atan(y / x) + Pi;
    else
      phi = std::atan(y / x) + Pi;
    phi = phi * 180.0 / Pi;

    return phi;
  }
}  // anonymous namespace

void SiStripHitEfficiencyWorker::bookHistograms(DQMStore::IBooker& booker,
                                                const edm::Run& run,
                                                const edm::EventSetup& setup) {
  const std::string path = "SiStrip/HitEfficiency"; // TODO make this configurable
  booker.setCurrentFolder(path);
  h_bx = booker.book1D("bx", "bx", 3600, 0, 3600);
  h_instLumi = booker.book1D("instLumi", "inst. lumi.", 250, 0, 25000);
  h_PU = booker.book1D("PU", "PU", 200, 0, 200);

  h_goodLayer = EffME1(
      booker.book1D("goodlayer_total", "goodlayer_total", 35, 0., 35.),
      booker.book1D("goodlayer_found", "goodlayer_found", 35, 0., 35.));
  h_allLayer = EffME1(
      booker.book1D("alllayer_total", "alllayer_total", 35, 0., 35.),
      booker.book1D("alllayer_found", "alllayer_found", 35, 0., 35.));

  h_layer = EffME1(
      booker.book1D("layer_found", "layer_found", 23, 0., 23.),
      booker.book1D("layer_total", "layer_total", 23, 0., 23.));
  for ( int iLayer = 0; iLayer != 23; ++iLayer ) {
    const auto lyrName = layerName(iLayer); // TODO change to std::string and {fmt}
    auto ihres = booker.book1D(Form("resol_layer_%i", iLayer), lyrName, 125, -125., 125.);
    ihres->setAxisTitle("trajX-clusX [strip unit]");
    h_resolution.push_back(ihres);
    h_layer_vsLumi.push_back(EffME1(
          booker.book1D(Form("layerfound_vsLumi_%i", iLayer), lyrName, 100, 0, 25000),
          booker.book1D(Form("layertotal_vsLumi_%i", iLayer), lyrName, 100, 0, 25000)));
    h_layer_vsPU.push_back(EffME1(
          booker.book1D(Form("layerfound_vsPU_%i", iLayer), lyrName, 45, 0, 90),
          booker.book1D(Form("layertotal_vsPU_%i", iLayer), lyrName, 45, 0, 90)));
    if (addCommonMode_) {
      h_layer_vsCM.push_back(EffME1(
            booker.book1D(Form("layerfound_vsCM_%i", iLayer), lyrName, 20, 0, 400),
            booker.book1D(Form("layertotal_vsCM_%i", iLayer), lyrName, 20, 0, 400)));
    }
    h_layer_vsBx.push_back(EffME1(
          booker.book1D(Form("totalVsBx_layer%i", iLayer), Form("layer %i", iLayer), 3565, 0, 3565),
          booker.book1D(Form("foundVsBx_layer%i", iLayer), Form("layer %i", iLayer), 3565, 0, 3565)));
    if ( iLayer < 10 ) {
      const bool isTIB = iLayer < 4;
      const auto partition = (isTIB ? "TIB" : "TOB");
      const auto yMax = (isTIB ? 100 : 120);
      auto ihhotcold = booker.book2D(
          Form("%s%i", partition, (isTIB ? iLayer+1 : iLayer-3)),
          partition, 100, -1, 361, 100, -yMax, yMax);
      ihhotcold->setAxisTitle("Phi", 1);
      ihhotcold->setBinLabel(1, "360", 1);
      ihhotcold->setBinLabel(50, "180", 1);
      ihhotcold->setBinLabel(100, "0", 1);
      ihhotcold->setAxisTitle("Global Z", 2);
      ihhotcold->setOption("colz");
      h_hotcold.push_back(ihhotcold);
    } else {
      const bool isTID = iLayer < 13;
      const auto partitions = (isTID ? std::vector<std::string>{"TID-", "TID+"} : std::vector<std::string>{"TEC-", "TEC+"});
      const auto axMax = (isTID ? 100 : 120);
      for ( const auto part : partitions ) {
        auto ihhotcold = booker.book2D(
            Form("%s%i", part.c_str(), (isTID ? iLayer-9 : iLayer-12)),
            part, 100, -axMax, axMax, 100, -axMax, axMax);
        ihhotcold->setAxisTitle("Global Y", 1);
        ihhotcold->setBinLabel(1, "+Y", 1);
        ihhotcold->setBinLabel(50, "0", 1);
        ihhotcold->setBinLabel(100, "-Y", 1);
        ihhotcold->setAxisTitle("Global X", 2);
        ihhotcold->setBinLabel(1, "-X", 2);
        ihhotcold->setBinLabel(50, "0", 2);
        ihhotcold->setBinLabel(100, "+X", 2);
        ihhotcold->setOption("colz");
        h_hotcold.push_back(ihhotcold);
      }
    }
  }

  edm::ESHandle<TrackerTopology> tTopoHandle;
  setup.get<TrackerTopologyRcd>().get(tTopoHandle);
  edm::ESHandle<TkDetMap> tkDetMapHandle;
  setup.get<TrackerTopologyRcd>().get(tkDetMapHandle);
  h_module = EffTkMap(
      std::make_unique<TkHistoMap>(tkDetMapHandle.product(), booker, path, "perModule_total", 0, false, true),
      std::make_unique<TkHistoMap>(tkDetMapHandle.product(), booker, path, "perModule_found", 0, false, true));
}

void SiStripHitEfficiencyWorker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //  bool DEBUG = false;

  LogDebug("SiStripHitEfficiency:HitEff") << "beginning analyze from HitEff" << std::endl;

  // Step A: Get Inputs

  // Luminosity informations
  edm::Handle<LumiScalersCollection> lumiScalers;
  float instLumi = 0;
  float PU = 0;
  if (addLumi_) {
    e.getByToken(scalerToken_, lumiScalers);
    if (lumiScalers->begin() != lumiScalers->end()) {
      instLumi = lumiScalers->begin()->instantLumi();
      PU = lumiScalers->begin()->pileup();
    }
  }
  h_bx->Fill(e.bunchCrossing());
  h_instLumi->Fill(instLumi);
  h_PU->Fill(PU);


  edm::Handle<edm::DetSetVector<SiStripRawDigi>> commonModeDigis;
  if (addCommonMode_)
    e.getByToken(commonModeToken_, commonModeDigis);

  edm::Handle<reco::TrackCollection> tracksCKF;
  e.getByToken(combinatorialTracks_token_, tracksCKF);

  edm::Handle<std::vector<Trajectory>> TrajectoryCollectionCKF;
  e.getByToken(trajectories_token_, TrajectoryCollectionCKF);

  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  e.getByToken(trajTrackAsso_token_, trajTrackAssociationHandle);

  edm::Handle<edmNew::DetSetVector<SiStripCluster>> theClusters;
  e.getByToken(clusters_token_, theClusters);

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  const TrackerGeometry* tkgeom = tracker.product();

  edm::ESHandle<StripClusterParameterEstimator> stripcpe;
  es.get<TkStripCPERecord>().get("StripCPEfromTrackAngle", stripcpe);

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  es.get<SiStripQualityRcd>().get(SiStripQuality_);

  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);

  edm::Handle<DetIdCollection> fedErrorIds;
  e.getByToken(digis_token_, fedErrorIds);

  edm::ESHandle<MeasurementTracker> measTracker;
  es.get<CkfComponentsRecord>().get(measTracker);

  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  e.getByToken(trackerEvent_token_, measurementTrackerEvent);

  edm::ESHandle<Chi2MeasurementEstimatorBase> estimator;
  es.get<TrackingComponentsRecord>().get("Chi2", estimator);

  edm::ESHandle<Propagator> prop;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", prop);

  ++events;

  // Tracking
  LogDebug("SiStripHitEfficiency:HitEff") << "number ckf tracks found = " << tracksCKF->size() << std::endl;
  if (!tracksCKF->empty()) {
    if (cutOnTracks_ && (tracksCKF->size() >= trackMultiplicityCut_))
      return;
    if (cutOnTracks_)
      LogDebug("SiStripHitEfficiency:HitEff")
          << "starting checking good event with < " << trackMultiplicityCut_ << " tracks" << std::endl;

    ++EventTrackCKF;

    // actually should do a loop over all the tracks in the event here

    // Looping over traj-track associations to be able to get traj & track informations
    for (const auto& trajTrack : *trajTrackAssociationHandle) {
      // for each track, fill some variables such as number of hits and momentum

      const bool highPurity = trajTrack.val->quality(reco::TrackBase::TrackQuality::highPurity);
      auto TMeas = trajTrack.key->measurements();

      const bool hasMissingHits = std::any_of(std::begin(TMeas), std::end(TMeas), [](const auto& tm) {
        return tm.recHit()->getType() == TrackingRecHit::Type::missing;
      });

      // Loop on each measurement and take it into consideration
      //--------------------------------------------------------
      for (auto itm = TMeas.cbegin(); itm != TMeas.cend(); ++itm) {
        const auto theInHit = (*itm).recHit();

        LogDebug("SiStripHitEfficiency:HitEff") << "theInHit is valid = " << theInHit->isValid() << std::endl;

        unsigned int iidd = theInHit->geographicalId().rawId();

        unsigned int TKlayers = checkLayer(iidd, tTopo);
        LogDebug("SiStripHitEfficiency:HitEff") << "TKlayer from trajectory: " << TKlayers << "  from module = " << iidd
                                                << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                                                << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2) << std::endl;

        // Test first and last points of the trajectory
        // the list of measurements starts from outer layers  !!! This could change -> should add a check
        if ((!useFirstMeas_ && (itm == (TMeas.end() - 1))) || (!useLastMeas_ && (itm == (TMeas.begin()))) ||
            // In case of missing hit in the track, check whether to use the other hits or not.
            (!useAllHitsFromTracksWithMissingHits_ && hasMissingHits &&
             theInHit->getType() != TrackingRecHit::Type::missing))
          continue;
        // If Trajectory measurement from TOB 6 or TEC 9, skip it because it's always valid they are filled later
        if (TKlayers == 10 || TKlayers == 22) {
          LogDebug("SiStripHitEfficiency:HitEff") << "skipping original TM for TOB 6 or TEC 9" << std::endl;
          continue;
        }

        std::vector<TrajectoryAtInvalidHit> TMs;

        // Make AnalyticalPropagat // TODO where to save these?or to use in TAVH constructor
        AnalyticalPropagator propagator(magField.product(), anyDirection);

        // for double sided layers check both sensors--if no hit was found on either sensor surface,
        // the trajectory measurements only have one invalid hit entry on the matched surface
        // so get the TrajectoryAtInvalidHit for both surfaces and include them in the study
        if (isDoubleSided(iidd, tTopo) && ((iidd & 0x3) == 0)) {
          // do hit eff check twice--once for each sensor
          //add a TM for each surface
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 1);
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 2);
        } else if (isDoubleSided(iidd, tTopo) && (!check2DPartner(iidd, TMeas))) {
          // if only one hit was found the trajectory measurement is on that sensor surface, and the other surface from
          // the matched layer should be added to the study as well
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 1);
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 2);
          LogDebug("SiStripHitEfficiency:HitEff") << " found a hit with a missing partner" << std::endl;
        } else {
          //only add one TM for the single surface and the other will be added in the next iteration
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator);
        }

        //////////////////////////////////////////////
        //Now check for tracks at TOB6 and TEC9

        // to make sure we only propagate on the last TOB5 hit check the next entry isn't also in TOB5
        // to avoid bias, make sure the TOB5 hit is valid (an invalid hit on TOB5 could only exist with a valid hit on TOB6)
        const auto nextId = (itm + 1 != TMeas.end()) ? (itm + 1)->recHit()->geographicalId() : DetId{};  // null if last

        if (TKlayers == 9 && theInHit->isValid() && !((!nextId.null()) && (checkLayer(nextId.rawId(), tTopo) == 9))) {
          //	  if ( TKlayers==9 && itm==TMeas.rbegin()) {
          //	  if ( TKlayers==9 && (itm==TMeas.back()) ) {	  // to check for only the last entry in the trajectory for propagation
          const DetLayer* tob6 = measTracker->geometricSearchTracker()->tobLayers().back();
          const LayerMeasurements theLayerMeasurements{*measTracker, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTOB5 = itm->updatedState();
          const auto tmp = theLayerMeasurements.measurements(*tob6, tsosTOB5, *prop, *estimator);

          if (!tmp.empty()) {
            LogDebug("SiStripHitEfficiency:HitEff") << "size of TM from propagation = " << tmp.size() << std::endl;

            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            const auto& tob6TM = tmp.back();
            const auto& tob6Hit = tob6TM.recHit();
            if (tob6Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tob6 hit actually being added to TM vector" << std::endl;
              TMs.emplace_back(tob6TM, tTopo, tkgeom, propagator);
            }
          }
        }

        // same for TEC8
        if (TKlayers == 21 && theInHit->isValid() && !((!nextId.null()) && (checkLayer(nextId.rawId(), tTopo) == 21))) {
          const DetLayer* tec9pos = measTracker->geometricSearchTracker()->posTecLayers().back();
          const DetLayer* tec9neg = measTracker->geometricSearchTracker()->negTecLayers().back();

          const LayerMeasurements theLayerMeasurements{*measTracker, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTEC9 = itm->updatedState();

          // check if track on positive or negative z
          if (!(iidd == SiStripSubdetector::TEC))
            LogDebug("SiStripHitEfficiency:HitEff") << "there is a problem with TEC 9 extrapolation" << std::endl;

          //cout << " tec9 id = " << iidd << " and side = " << tTopo->tecSide(iidd) << std::endl;
          std::vector<TrajectoryMeasurement> tmp;
          if (tTopo->tecSide(iidd) == 1) {
            tmp = theLayerMeasurements.measurements(*tec9neg, tsosTEC9, *prop, *estimator);
            //cout << "on negative side" << std::endl;
          }
          if (tTopo->tecSide(iidd) == 2) {
            tmp = theLayerMeasurements.measurements(*tec9pos, tsosTEC9, *prop, *estimator);
            //cout << "on positive side" << std::endl;
          }

          if (!tmp.empty()) {
            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            const auto& tec9TM = tmp.back();
            const auto& tec9Hit = tec9TM.recHit();

            const unsigned int tec9id = tec9Hit->geographicalId().rawId();
            LogDebug("SiStripHitEfficiency:HitEff")
                << "tec9id = " << tec9id << " is Double sided = " << isDoubleSided(tec9id, tTopo)
                << "  and 0x3 = " << (tec9id & 0x3) << std::endl;

            if (tec9Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tec9 hit actually being added to TM vector" << std::endl;
              // in tec the hit can be single or doubled sided. whenever the invalid hit at the end of vector of TMs is
              // double sided it is always on the matched surface, so we need to split it into the true sensor surfaces
              if (isDoubleSided(tec9id, tTopo)) {
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator, 1);
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator, 2);
              } else
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator);
            }
          }  //else std::cout << "tec9 tmp empty" << std::endl;
        }

        for (const auto& tm : TMs) {
          fillForTraj(tm,
                      tTopo,
                      tkgeom,
                      stripcpe.product(),
                      SiStripQuality_.product(),
                      *fedErrorIds,
                      commonModeDigis,
                      *theClusters,
                      e.bunchCrossing(),
                      instLumi,
                      PU,
                      highPurity);
        }
        LogDebug("SiStripHitEfficiency:HitEff") << "After looping over TrajAtValidHit list" << std::endl;
      }
      LogDebug("SiStripHitEfficiency:HitEff") << "end TMeasurement loop" << std::endl;
    }
    LogDebug("SiStripHitEfficiency:HitEff") << "end of trajectories loop" << std::endl;
  }
}

TString SiStripHitEfficiencyWorker::layerName(unsigned int k) const {
  auto ringlabel = _showRings ? TString("R") : TString("D");
  if (k > 0 && k < 5) {
    return TString("TIB L") + k;
  } else if (k > 4 && k < 11) {
    return TString("TOB L") + (k - 4);
  } else if (k > 10 && k < 14) {
    return TString("TID ") + ringlabel + (k - 10);
  } else if (k > 13 && k < 14 + (_showRings ? 7 : 9)) {
    return TString("TEC ") + ringlabel + (k - 13);
  }
  return "";
}

void SiStripHitEfficiencyWorker::fillForTraj(const TrajectoryAtInvalidHit& tm,
                                             const TrackerTopology* tTopo,
                                             const TrackerGeometry* tkgeom,
                                             const StripClusterParameterEstimator* stripCPE,
                                             const SiStripQuality* stripQuality,
                                             const DetIdCollection& fedErrorIds,
                                             const edm::Handle<edm::DetSetVector<SiStripRawDigi>>& commonModeDigis,
                                             const edmNew::DetSetVector<SiStripCluster>& theClusters,
                                             int bunchCrossing,
                                             float instLumi,
                                             float PU,
                                             bool highPurity
                                             ) {
  // --> Get trajectory from combinatedStat& e
  const auto iidd = tm.monodet_id();
  LogDebug("SiStripHitEfficiency:HitEff") << "setting iidd = " << iidd << " before checking efficiency and ";

  const auto xloc = tm.localX();
  const auto yloc = tm.localY();

  const auto xErr = tm.localErrorX();
  const auto yErr = tm.localErrorY();

  int TrajStrip = -1;

  // reget layer from iidd here, to account for TOB 6 and TEC 9 TKlayers being off
  const auto TKlayers = checkLayer(iidd, tTopo);

  const bool withinAcceptance = tm.withinAcceptance() && (!isInBondingExclusionZone(iidd, TKlayers, yloc, yErr, tTopo));

  if ((layers == TKlayers) || (layers == 0)) {  // Look at the layer not used to reconstruct the track
    LogDebug("SiStripHitEfficiency:HitEff") << "Looking at layer under study" << std::endl;
    unsigned int ModIsBad = 2;
    unsigned int SiStripQualBad = 0;
    float commonMode = -100;

    // RPhi RecHit Efficiency

    if (!theClusters.empty()) {
      LogDebug("SiStripHitEfficiency:HitEff") << "Checking clusters with size = " << theClusters.size() << std::endl;
      std::vector<ClusterInfo> VCluster_info;  //fill with X residual, X residual pull, local X
      const auto idsv = theClusters.find(iidd);
      if (idsv != theClusters.end()) {
        //if (DEBUG)      std::cout << "the ID from the dsv = " << dsv.id() << std::endl;
        LogDebug("SiStripHitEfficiency:HitEff")
            << "found  (ClusterId == iidd) with ClusterId = " << idsv->id() << " and iidd = " << iidd << std::endl;
        const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(iidd)));
        const StripTopology& Topo = stripdet->specificTopology();

        float hbedge = 0.0;
        float htedge = 0.0;
        float hapoth = 0.0;
        float uylfac = 0.0;
        float uxlden = 0.0;
        if (TKlayers >= 11) {
          const BoundPlane& plane = stripdet->surface();
          const TrapezoidalPlaneBounds* trapezoidalBounds(
              dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
          std::array<const float, 4> const& parameterTrap = (*trapezoidalBounds).parameters();  // el bueno aqui
          hbedge = parameterTrap[0];
          htedge = parameterTrap[1];
          hapoth = parameterTrap[3];
          uylfac = (htedge - hbedge) / (htedge + hbedge) / hapoth;
          uxlden = 1 + yloc * uylfac;
        }

        // Need to know position of trajectory in strip number for selecting the right APV later
        if (TrajStrip == -1) {
          int nstrips = Topo.nstrips();
          float pitch = stripdet->surface().bounds().width() / nstrips;
          TrajStrip = xloc / pitch + nstrips / 2.0;
          // Need additionnal corrections for endcap
          if (TKlayers >= 11) {
            const float TrajLocXMid = xloc / (1 + (htedge - hbedge) * yloc / (htedge + hbedge) /
                                                hapoth);  // radialy extrapolated x loc position at middle
            TrajStrip = TrajLocXMid / pitch + nstrips / 2.0;
          }
          //cout<<" Layer "<<TKlayers<<" TrajStrip: "<<nstrips<<" "<<pitch<<" "<<TrajStrip<<endl;
        }

        for (const auto& clus : *idsv) {
          StripClusterParameterEstimator::LocalValues parameters = stripCPE->localParameters(clus, *stripdet);
          float res = (parameters.first.x() - xloc);
          float sigma = checkConsistency(parameters, xloc, xErr);
          // The consistency is probably more accurately measured with the Chi2MeasurementEstimator. To use it
          // you need a TransientTrackingRecHit instead of the cluster
          //theEstimator=       new Chi2MeasurementEstimator(30);
          //const Chi2MeasurementEstimator *theEstimator(100);
          //theEstimator->estimate(tm.tsos(), TransientTrackingRecHit);

          if (TKlayers >= 11) {
            res = parameters.first.x() - xloc / uxlden;  // radialy extrapolated x loc position at middle
            sigma = abs(res) / sqrt(parameters.second.xx() + xErr * xErr / uxlden / uxlden +
                                    yErr * yErr * xloc * xloc * uylfac * uylfac / uxlden / uxlden / uxlden / uxlden);
          }

          VCluster_info.emplace_back(res, sigma, parameters.first.x());

          LogDebug("SiStripHitEfficiency:HitEff")
              << "Have ID match. residual = " << res << "  res sigma = " << sigma << std::endl;
          //LogDebug("SiStripHitEfficiency:HitEff")
          //    << "trajectory measurement compatability estimate = " << (*itm).estimate() << std::endl;
          LogDebug("SiStripHitEfficiency:HitEff")
              << "hit position = " << parameters.first.x() << "  hit error = " << sqrt(parameters.second.xx())
              << "  trajectory position = " << xloc << "  traj error = " << xErr << std::endl;
        }
      }
      ClusterInfo finalCluster{1000.0, 1000.0, 0.0};
      if (!VCluster_info.empty()) {
        LogDebug("SiStripHitEfficiency:HitEff") << "found clusters > 0" << std::endl;
        if (VCluster_info.size() > 1) {
          //get the smallest one
          for (const auto& res : VCluster_info) {
            if (std::abs(res.xResidualPull) < std::abs(finalCluster.xResidualPull)) {
              finalCluster = res;
            }
            LogDebug("SiStripHitEfficiency:HitEff")
                << "iresidual = " << res.xResidual << "  isigma = " << res.xResidualPull
                << "  and FinalRes = " << finalCluster.xResidual << std::endl;
          }
        } else {
          finalCluster = VCluster_info[0];
        }
        VCluster_info.clear();
      }

      LogDebug("SiStripHitEfficiency:HitEff") << "Final residual in X = " << finalCluster.xResidual << "+-"
                                              << (finalCluster.xResidual / finalCluster.xResidualPull) << std::endl;
      LogDebug("SiStripHitEfficiency:HitEff")
          << "Checking location of trajectory: abs(yloc) = " << abs(yloc) << "  abs(xloc) = " << abs(xloc) << std::endl;

      //
      // fill ntuple varibles

      //if ( SiStripQuality_->IsModuleBad(iidd) )
      if (stripQuality->getBadApvs(iidd) != 0) {
        SiStripQualBad = 1;
        LogDebug("SiStripHitEfficiency:HitEff") << "strip is bad from SiStripQuality" << std::endl;
      } else {
        SiStripQualBad = 0;
        LogDebug("SiStripHitEfficiency:HitEff") << "strip is good from SiStripQuality" << std::endl;
      }

      //check for FED-detected errors and include those in SiStripQualBad
      for (unsigned int ii = 0; ii < fedErrorIds.size(); ii++) {
        if (iidd == fedErrorIds[ii].rawId())
          SiStripQualBad = 1;
      }

      // CM of APV crossed by traj
      if (addCommonMode_)
        if (commonModeDigis.isValid() && TrajStrip >= 0 && TrajStrip <= 768) {
          const auto digiframe = commonModeDigis->find(iidd);
          if (digiframe != commonModeDigis->end())
            if ((unsigned)TrajStrip / 128 < digiframe->data.size())
              commonMode = digiframe->data.at(TrajStrip / 128).adc();
        }

      LogDebug("SiStripHitEfficiency:HitEff") << "before check good" << std::endl;

      if (finalCluster.xResidualPull < 999.0) {  //could make requirement on track/hit consistency, but for
        //now take anything with a hit on the module
        LogDebug("SiStripHitEfficiency:HitEff")
            << "hit being counted as good " << finalCluster.xResidual << " FinalRecHit " << iidd << "   TKlayers  "
            << TKlayers << " xloc " << xloc << " yloc  " << yloc << " module " << iidd
            << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1) << "/"
            << ((iidd & 0x3) == 2) << std::endl;
        ModIsBad = 0;
      } else {
        LogDebug("SiStripHitEfficiency:HitEff")
            << "hit being counted as bad   ######### Invalid RPhi FinalResX " << finalCluster.xResidual
            << " FinalRecHit " << iidd << "   TKlayers  " << TKlayers << " xloc " << xloc << " yloc  " << yloc
            << " module " << iidd << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1)
            << "/" << ((iidd & 0x3) == 2) << std::endl;
        ModIsBad = 1;
        LogDebug("SiStripHitEfficiency:HitEff")
            << " RPhi Error " << sqrt(xErr * xErr + yErr * yErr) << " ErrorX " << xErr << " yErr " << yErr << std::endl;
      }

      LogDebug("SiStripHitEfficiency:HitEff") << "To avoid them staying unused: ModIsBad=" << ModIsBad << ", SiStripQualBad=" << SiStripQualBad << ", commonMode=" << commonMode << ", highPurity=" << highPurity << ", withinAcceptance=" << withinAcceptance;

      unsigned int layer = TKlayers;
      if (_showRings && layer > 10) { // use rings instead of wheels
        if (layer < 14) { // TID
          layer = 10 + ((iidd >> 9) & 0x3); // 3 disks and also 3 rings -> use the same container
        } else { // TEC
          layer = 13 + ((iidd >> 5) & 0x7);
        }
      }
      unsigned int layerWithSide = layer;
      if ( layer > 10 && layer < 14 ) {
        const auto side = (iidd>>13)&0x3; // TID
        if ( side == 2 )
          layerWithSide = layer + 3;
      } else if ( layer > 13 ) {
        const auto side = (iidd>>18)&0x3; // TEC
        if ( side == 1 ) {
          layerWithSide = layer + 3;
        } else if ( side == 2 ) {
          layerWithSide = layer + 3 + (_showRings ? 7 : 9);
        }
      }

      if ( (_bunchx > 0 && _bunchx != bunchCrossing) ||
           (!withinAcceptance) ||
           (_useOnlyHighPurityTracks && !highPurity) ||
           (!_showTOB6TEC9 && (TKlayers == 10 || TKlayers == 22)) ||
           ( badModules_.end() != badModules_.find(iidd) ) )
        return;

      const bool badquality = (SiStripQualBad == 1);

      //Now that we have a good event, we need to look at if we expected it or not, and the location
      //if we didn't
      //Fill the missing hit information first
      bool badflag = false; // true for hits that are expected but not found
      if (_ResXSig < 0) {
        if (ModIsBad == 1)
          badflag = true;  // isBad set to false in the tree when resxsig<999.0
      } else {
        if (ModIsBad == 1 || finalCluster.xResidualPull > _ResXSig)
          badflag = true;
      }

      // Conversion of positions in strip unit
      int nstrips = -9;
      float Pitch = -9.0;
      const StripGeomDetUnit* stripdet = nullptr;
      if (finalCluster.xResidualPull == 1000.0) {  // special treatment, no GeomDetUnit associated in some cases when no cluster found
        Pitch = 0.0205;         // maximum
        nstrips = 768;          // maximum
      } else {
        stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(iidd));
        const StripTopology& Topo = stripdet->specificTopology();
        nstrips = Topo.nstrips();
        Pitch = stripdet->surface().bounds().width() / Topo.nstrips();
      }
      double stripTrajMid = xloc / Pitch + nstrips / 2.0;
      double stripCluster = finalCluster.xLocal / Pitch + nstrips / 2.0;
      // For trapezoidal modules: extrapolation of x trajectory position to the y middle of the module
      //  for correct comparison with cluster position
      if (stripdet && layer >= 11) {
        const auto& trapezoidalBounds = dynamic_cast<const TrapezoidalPlaneBounds&>(stripdet->surface().bounds());
        std::array<const float, 4> const& parameters = trapezoidalBounds.parameters();
        const float hbedge = parameters[0];
        const float htedge = parameters[1];
        const float hapoth = parameters[3];
        const float TrajLocXMid = xloc / (1 + (htedge - hbedge) * yloc / (htedge + hbedge) /
                                          hapoth);  // radialy extrapolated x loc position at middle
        stripTrajMid = TrajLocXMid / Pitch + nstrips / 2.0;
      }

      if ( (!badquality) && (layer < h_resolution.size())) {
        h_resolution[layer]->Fill(finalCluster.xResidualPull != 1000.0 ?
            stripTrajMid-stripCluster : 1000);
      }

      // New matching methods
      if (_clusterMatchingMethod >= 1) {
        badflag = false;
        if ( finalCluster.xResidualPull == 1000.0 ) {
          badflag = true;
        } else {
          if (_clusterMatchingMethod == 2 || _clusterMatchingMethod == 4) {
            // check the distance between cluster and trajectory position
            if (std::abs(stripCluster - stripTrajMid) > _clusterTrajDist)
              badflag = true;
          }
          if (_clusterMatchingMethod == 3 || _clusterMatchingMethod == 4) {
            // cluster and traj have to be in the same APV (don't take edges into accounts)
            const int tapv = (int)stripTrajMid / 128;
            const int capv = (int)stripTrajMid / 128;
            float stripInAPV = stripTrajMid - tapv * 128;
            if (stripInAPV < _stripsApvEdge || stripInAPV > 128 - _stripsApvEdge)
              return;
            if (tapv != capv)
              badflag = true;
          }
        }
      }
      if (!badquality) {
        // hot/cold maps of hits that are expected but not found
        if (badflag) {
          if ( layer > 0 && layer <= 10 ) {
            // 1-4: TIB, 4-10: TOB
            h_hotcold[layer-1]->Fill(360.-calcPhi(tm.globalX(), tm.globalY()), tm.globalZ(), 1.);
          } else if ( layer > 10 && layer <= 13 ) {
            // 11-13: TID, above: TEC
            const int side = layer > 13 ? (iidd>>13)&0x3 : (iidd>>18)&0x3;
            h_hotcold[2*layer-13+side]->Fill(-tm.globalY(), tm.globalX(), 1.);
          }
        }

        // efficiency with bad modules excluded
        h_module.fill(iidd, !badflag);
        h_layer_vsBx[layer].fill(bunchCrossing, !badflag);
        if (addLumi_) {
          h_layer_vsLumi[layer].fill(instLumi, !badflag);
          h_layer_vsPU[layer].fill(PU, !badflag);
        }
        if (addCommonMode_) {
          h_layer_vsCM[layer].fill(commonMode, !badflag);
        }
        h_goodLayer.fill(layerWithSide, !badflag);
      }
      // efficiency without bad modules excluded
      h_allLayer.fill(layerWithSide, !badflag);

      /* Used in SiStripHitEffFromCalibTree:
* run              -> "run"              -> run              // e.id().run()
* event            -> "event"            -> evt              // e.id().event()
* ModIsBad         -> "ModIsBad"         -> isBad
* SiStripQualBad   -> "SiStripQualBad""  -> quality
* Id               -> "Id"               -> id               // iidd
* withinAcceptance -> "withinAcceptance" -> accept
* whatlayer        -> "layer"            -> layer_wheel      // Tklayers
* highPurity       -> "highPurity"       -> highPurity
* TrajGlbX         -> "TrajGlbX"         -> x                // tm.globalX()
* TrajGlbY         -> "TrajGlbY"         -> y                // tm.globalY()
* TrajGlbZ         -> "TrajGlbZ"         -> z                // tm.globalZ()
* ResXSig          -> "ResXSig"          -> resxsig          // finalCluster.xResidualPull;
* TrajLocX         -> "TrajLocX"         -> TrajLocX         // xloc
* TrajLocY         -> "TrajLocY"         -> TrajLocY         // yloc
* ClusterLocX      -> "ClusterLocX"      -> ClusterLocX      // finalCluster.xLocal
* bunchx           -> "bunchx"           -> bx               // e.bunchCrossing()
* instLumi         -> "instLumi"         -> instLumi         ## if addLumi_
* PU               -> "PU"               -> PU               ## if addLumi_
* commonMode       -> "commonMode"       -> CM               ## if addCommonMode_ / _useCM
*/
      LogDebug("SiStripHitEfficiency:HitEff") << "after good location check" << std::endl;
    }
    LogDebug("SiStripHitEfficiency:HitEff") << "after list of clusters" << std::endl;
  }
  LogDebug("SiStripHitEfficiency:HitEff") << "After layers=TKLayers if" << std::endl;
}

void SiStripHitEfficiencyWorker::endJob() {
  LogDebug("SiStripHitEfficiency:HitEff") << " Events Analysed             " << events << std::endl;
  LogDebug("SiStripHitEfficiency:HitEff") << " Number Of Tracked events    " << EventTrackCKF << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyWorker);

// TODO next: try to run this
