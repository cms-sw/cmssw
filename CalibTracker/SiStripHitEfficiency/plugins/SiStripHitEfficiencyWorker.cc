////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitEfficiency
// Class:            SiStripHitEfficiencyWorker
// Original Author:  Pieter David
//
// Adapted from      HitEff (Keith Ulmer -- University of Colorado, keith.ulmer@colorado.edu
//                   SiStripHitEffFromCalibTree (Christopher Edelmaier)
///////////////////////////////////////////////////////////////////////////////

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEffData.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEfficiencyHelpers.h"
#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h" /* for STRIPS_PER_APV*/
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class SiStripHitEfficiencyWorker : public DQMEDAnalyzer {
public:
  explicit SiStripHitEfficiencyWorker(const edm::ParameterSet& conf);
  ~SiStripHitEfficiencyWorker() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker& booker, const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void fillForTraj(const TrajectoryAtInvalidHit& tm,
                   const TrackerTopology* tTopo,
                   const TrackerGeometry* tkgeom,
                   const StripClusterParameterEstimator& stripCPE,
                   const SiStripQuality& stripQuality,
                   const DetIdCollection& fedErrorIds,
                   const edm::Handle<edm::DetSetVector<SiStripRawDigi>>& commonModeDigis,
                   const edmNew::DetSetVector<SiStripCluster>& theClusters,
                   int bunchCrossing,
                   float instLumi,
                   float PU,
                   bool highPurity);

  // ----------member data ---------------------------
  SiStripHitEffData calibData_;

  // event data tokens
  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>> commonModeToken_;
  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory>> trajectories_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAsso_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digis_token_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> trackerEvent_token_;

  // event setup tokens
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> stripCPEToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measTrackerToken_;
  const edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> chi2EstimatorToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  const edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;

  // configurable parameters
  std::string dqmDir_;
  unsigned int layers_;
  bool DEBUG_;
  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_;
  bool doMissingHitsRecovery_;
  unsigned int clusterMatchingMethod_;
  float resXSig_;
  float clusterTracjDist_;
  float stripsApvEdge_;
  bool useOnlyHighPurityTracks_;
  int bunchX_;
  bool showRings_;
  bool showTOB6TEC9_;
  unsigned int nTEClayers_;

  // output file
  std::set<uint32_t> badModules_;

  // for the missing hits recovery
  std::vector<unsigned int> hitRecoveryCounters;
  std::vector<unsigned int> hitTotalCounters;
  int totalNbHits;
  std::vector<int> missHitPerLayer;

  struct EffME1 {
    EffME1() : hTotal(nullptr), hFound(nullptr) {}
    EffME1(MonitorElement* total, MonitorElement* found) : hTotal(total), hFound(found) {}

    void fill(double x, bool found, float weight = 1.) {
      hTotal->Fill(x, weight);
      if (found) {
        hFound->Fill(x, weight);
      }
    }

    MonitorElement *hTotal, *hFound;
  };
  struct EffTkMap {
    EffTkMap() : hTotal(nullptr), hFound(nullptr) {}
    EffTkMap(std::unique_ptr<TkHistoMap>&& total, std::unique_ptr<TkHistoMap>&& found)
        : hTotal(std::move(total)), hFound(std::move(found)) {}

    void fill(uint32_t id, bool found, float weight = 1.) {
      hTotal->fill(id, weight);
      if (found) {
        hFound->fill(id, weight);
      }
    }

    bool check(uint32_t id) {
      if (hTotal->getValue(id) < hFound->getValue(id)) {
        return false;
      } else {
        return true;
      }
    }

    std::unique_ptr<TkHistoMap> hTotal, hFound;
  };

  MonitorElement *h_bx, *h_instLumi, *h_PU;
  MonitorElement *h_nTracks, *h_nTracksVsPU;
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
      metaDataToken_(consumes<OnlineLuminosityRecord>(conf.getParameter<edm::InputTag>("metadata"))),
      commonModeToken_(mayConsume<edm::DetSetVector<SiStripRawDigi>>(conf.getParameter<edm::InputTag>("commonMode"))),
      combinatorialTracks_token_(
          consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("combinatorialTracks"))),
      trajectories_token_(consumes<std::vector<Trajectory>>(conf.getParameter<edm::InputTag>("trajectories"))),
      trajTrackAsso_token_(consumes<TrajTrackAssociationCollection>(conf.getParameter<edm::InputTag>("trajectories"))),
      clusters_token_(
          consumes<edmNew::DetSetVector<SiStripCluster>>(conf.getParameter<edm::InputTag>("siStripClusters"))),
      digis_token_(consumes<DetIdCollection>(conf.getParameter<edm::InputTag>("siStripDigis"))),
      trackerEvent_token_(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))),
      tTopoToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      stripCPEToken_(esConsumes(edm::ESInputTag{"", "StripCPEfromTrackAngle"})),
      stripQualityToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      measTrackerToken_(esConsumes()),
      chi2EstimatorToken_(esConsumes(edm::ESInputTag{"", "Chi2"})),
      propagatorToken_(esConsumes(edm::ESInputTag{"", "PropagatorWithMaterial"})),
      tkDetMapToken_(esConsumes<edm::Transition::BeginRun>()),
      dqmDir_(conf.getParameter<std::string>("dqmDir")),
      layers_(conf.getParameter<int>("Layer")),
      DEBUG_(conf.getUntrackedParameter<bool>("Debug", false)),
      addLumi_(conf.getUntrackedParameter<bool>("addLumi", false)),
      addCommonMode_(conf.getUntrackedParameter<bool>("addCommonMode", false)),
      cutOnTracks_(conf.getParameter<bool>("cutOnTracks")),
      trackMultiplicityCut_(conf.getParameter<unsigned int>("trackMultiplicity")),
      useFirstMeas_(conf.getParameter<bool>("useFirstMeas")),
      useLastMeas_(conf.getParameter<bool>("useLastMeas")),
      useAllHitsFromTracksWithMissingHits_(conf.getParameter<bool>("useAllHitsFromTracksWithMissingHits")),
      doMissingHitsRecovery_(conf.getParameter<bool>("doMissingHitsRecovery")),
      clusterMatchingMethod_(conf.getParameter<int>("ClusterMatchingMethod")),
      resXSig_(conf.getParameter<double>("ResXSig")),
      clusterTracjDist_(conf.getParameter<double>("ClusterTrajDist")),
      stripsApvEdge_(conf.getParameter<double>("StripsApvEdge")),
      useOnlyHighPurityTracks_(conf.getParameter<bool>("UseOnlyHighPurityTracks")),
      bunchX_(conf.getUntrackedParameter<int>("BunchCrossing", 0)),
      showRings_(conf.getUntrackedParameter<bool>("ShowRings", false)),
      showTOB6TEC9_(conf.getUntrackedParameter<bool>("ShowTOB6TEC9", false)) {
  hitRecoveryCounters.resize(k_END_OF_LAYERS, 0);
  hitTotalCounters.resize(k_END_OF_LAYERS, 0);
  missHitPerLayer.resize(k_END_OF_LAYERS, 0);
  totalNbHits = 0;

  nTEClayers_ = (showRings_ ? 7 : 9);  // number of rings or wheels

  const std::string badModulesFile = conf.getUntrackedParameter<std::string>("BadModulesFile", "");
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
    LogDebug("SiStripHitEfficiencyWorker") << "Remove additionnal bad modules from the analysis: ";
  for (const auto badMod : badModules_) {
    LogDebug("SiStripHitEfficiencyWorker") << " " << badMod;
  }
}

void SiStripHitEfficiencyWorker::bookHistograms(DQMStore::IBooker& booker,
                                                const edm::Run& run,
                                                const edm::EventSetup& setup) {
  booker.setCurrentFolder(fmt::format("{}/EventInfo", dqmDir_));
  h_bx = booker.book1D("bx", "bx", 3600, 0, 3600);
  h_instLumi = booker.book1D("instLumi", "inst. lumi.", 250, 0, 25000);
  h_PU = booker.book1D("PU", "PU", 200, 0, 200);
  h_nTracks = booker.book1D("ntracks", "n.tracks;n. tracks;n.events", 500, -0.5, 499.5);
  h_nTracksVsPU = booker.bookProfile("nTracksVsPU", "n. tracks vs PU; PU; n.tracks ", 200, 0, 200, 500, -0.5, 499.5);

  calibData_.EventStats = booker.book2I("EventStats", "Statistics", 3, -0.5, 2.5, 1, 0, 1);
  calibData_.EventStats->setBinLabel(1, "events count", 1);
  calibData_.EventStats->setBinLabel(2, "tracks count", 1);
  calibData_.EventStats->setBinLabel(3, "measurements count", 1);

  booker.setCurrentFolder(dqmDir_);
  h_goodLayer = EffME1(booker.book1D("goodlayer_total", "goodlayer_total", 35, 0., 35.),
                       booker.book1D("goodlayer_found", "goodlayer_found", 35, 0., 35.));
  h_allLayer = EffME1(booker.book1D("alllayer_total", "alllayer_total", 35, 0., 35.),
                      booker.book1D("alllayer_found", "alllayer_found", 35, 0., 35.));

  h_layer = EffME1(
      booker.book1D(
          "layer_found", "layer_found", bounds::k_END_OF_LAYERS, 0., static_cast<float>(bounds::k_END_OF_LAYERS)),
      booker.book1D(
          "layer_total", "layer_total", bounds::k_END_OF_LAYERS, 0., static_cast<float>(bounds::k_END_OF_LAYERS)));

  for (int layer = 1; layer != bounds::k_END_OF_LAYERS; ++layer) {
    const auto lyrName = ::layerName(layer, showRings_, nTEClayers_);

    // book resolutions
    booker.setCurrentFolder(fmt::format("{}/Resolutions", dqmDir_));
    auto ihres = booker.book1D(Form("resol_layer_%i", layer), lyrName, 125, -125., 125.);
    ihres->setAxisTitle("trajX-clusX [strip unit]");
    h_resolution.push_back(ihres);

    // book plots vs Lumi
    booker.setCurrentFolder(fmt::format("{}/VsLumi", dqmDir_));
    h_layer_vsLumi.push_back(EffME1(booker.book1D(Form("layertotal_vsLumi_layer_%i", layer), lyrName, 100, 0, 25000),
                                    booker.book1D(Form("layerfound_vsLumi_layer_%i", layer), lyrName, 100, 0, 25000)));

    // book plots vs Lumi
    booker.setCurrentFolder(fmt::format("{}/VsPu", dqmDir_));
    h_layer_vsPU.push_back(EffME1(booker.book1D(Form("layertotal_vsPU_layer_%i", layer), lyrName, 45, 0, 90),
                                  booker.book1D(Form("layerfound_vsPU_layer_%i", layer), lyrName, 45, 0, 90)));
    if (addCommonMode_) {
      // book plots for common mode
      booker.setCurrentFolder(fmt::format("{}/CommonMode", dqmDir_));
      h_layer_vsCM.push_back(EffME1(booker.book1D(Form("layertotal_vsCM_layer_%i", layer), lyrName, 20, 0, 400),
                                    booker.book1D(Form("layerfound_vsCM_layer_%i", layer), lyrName, 20, 0, 400)));
    }

    // book plots vs Lumi
    booker.setCurrentFolder(fmt::format("{}/VsBx", dqmDir_));
    h_layer_vsBx.push_back(EffME1(
        booker.book1D(Form("totalVsBx_layer%i", layer), Form("layer %i (%s)", layer, lyrName.c_str()), 3565, 0, 3565),
        booker.book1D(Form("foundVsBx_layer%i", layer), Form("layer %i (%s)", layer, lyrName.c_str()), 3565, 0, 3565)));

    // book hot and cold
    booker.setCurrentFolder(fmt::format("{}/MissingHits", dqmDir_));
    if (layer <= bounds::k_LayersAtTOBEnd) {
      const bool isTIB = layer <= bounds::k_LayersAtTIBEnd;
      const auto partition = (isTIB ? "TIB" : "TOB");
      const auto yMax = (isTIB ? 100 : 120);

      const auto& tit =
          Form("%s%i: Map of missing hits", partition, (isTIB ? layer : layer - bounds::k_LayersAtTIBEnd));

      // histogram name must not contain ":" otherwise it fails upload to the GUI
      // see https://github.com/cms-DQM/dqmgui_prod/blob/af0a388e8f57c60e51111585d298aeeea943367f/src/cpp/DQM/DQMStore.cc#L56
      std::string name{tit};
      ::replaceInString(name, ":", "");

      auto ihhotcold = booker.book2D(name, tit, 100, -1, 361, 100, -yMax, yMax);
      ihhotcold->setAxisTitle("#phi [deg]", 1);
      ihhotcold->setBinLabel(1, "360", 1);
      ihhotcold->setBinLabel(50, "180", 1);
      ihhotcold->setBinLabel(100, "0", 1);
      ihhotcold->setAxisTitle("Global Z [cm]", 2);
      ihhotcold->setOption("colz");
      h_hotcold.push_back(ihhotcold);
    } else {
      const bool isTID = layer <= bounds::k_LayersAtTIDEnd;
      const auto partitions =
          (isTID ? std::vector<std::string>{"TIDplus", "TIDminus"} : std::vector<std::string>{"TECplus", "TECminus"});
      const auto axMax = (isTID ? 100 : 120);
      for (const auto& part : partitions) {
        // create the title by replacing the minus/plus symbols
        std::string forTitle{part};
        ::replaceInString(forTitle, "minus", "-");
        ::replaceInString(forTitle, "plus", "+");

        // histogram name must not contain ":" otherwise it fails upload to the GUI
        // see https://github.com/cms-DQM/dqmgui_prod/blob/af0a388e8f57c60e51111585d298aeeea943367f/src/cpp/DQM/DQMStore.cc#L56
        const auto& name = Form("%s %i Map of Missing Hits",
                                part.c_str(),
                                (isTID ? layer - bounds::k_LayersAtTOBEnd : layer - bounds::k_LayersAtTIDEnd));
        const auto& tit = Form("%s%i: Map of Missing Hits",
                               forTitle.c_str(),
                               (isTID ? layer - bounds::k_LayersAtTOBEnd : layer - bounds::k_LayersAtTIDEnd));

        auto ihhotcold = booker.book2D(name, tit, 100, -axMax, axMax, 100, -axMax, axMax);
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

  // come back to the main folder
  booker.setCurrentFolder(dqmDir_);
  const auto tkDetMapFolder = fmt::format("{}/TkDetMaps", dqmDir_);

  const TkDetMap* tkDetMap = &setup.getData(tkDetMapToken_);
  h_module =
      EffTkMap(std::make_unique<TkHistoMap>(tkDetMap, booker, tkDetMapFolder, "perModule_total", 0, false, true),
               std::make_unique<TkHistoMap>(tkDetMap, booker, tkDetMapFolder, "perModule_found", 0, false, true));

  // fill the FED Errors
  booker.setCurrentFolder(dqmDir_);
  const auto FEDErrorMapFolder = fmt::format("{}/FEDErrorTkDetMaps", dqmDir_);
  calibData_.FEDErrorOccupancy =
      std::make_unique<TkHistoMap>(tkDetMap, booker, FEDErrorMapFolder, "perModule_FEDErrors", 0, false, true);
}

void SiStripHitEfficiencyWorker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  const auto tTopo = &es.getData(tTopoToken_);

  //  bool DEBUG_ = false;

  LogDebug("SiStripHitEfficiencyWorker") << "beginning analyze from HitEff";

  // Step A: Get Inputs

  // Luminosity informations
  edm::Handle<LumiScalersCollection> lumiScalers = e.getHandle(scalerToken_);
  edm::Handle<OnlineLuminosityRecord> metaData = e.getHandle(metaDataToken_);

  float instLumi = 0;
  float PU = 0;
  if (addLumi_) {
    if (lumiScalers.isValid() && !lumiScalers->empty()) {
      if (lumiScalers->begin() != lumiScalers->end()) {
        instLumi = lumiScalers->begin()->instantLumi();
        PU = lumiScalers->begin()->pileup();
      }
    } else if (metaData.isValid()) {
      instLumi = metaData->instLumi();
      PU = metaData->avgPileUp();
    } else {
      edm::LogWarning("SiStripHitEfficiencyWorker") << "could not find a source for the Luminosity and PU";
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

  edm::Handle<DetIdCollection> fedErrorIds;
  e.getByToken(digis_token_, fedErrorIds);

  // fill the calibData with the FEDErrors
  for (const auto& fedErr : *fedErrorIds) {
    // fill the TkHistoMap occupancy map
    calibData_.FEDErrorOccupancy->fill(fedErr.rawId(), 1.);

    // fill the unordered map
    if (calibData_.fedErrorCounts.find(fedErr.rawId()) != calibData_.fedErrorCounts.end()) {
      calibData_.fedErrorCounts[fedErr.rawId()] += 1;
    } else {
      calibData_.fedErrorCounts.insert(std::make_pair(fedErr.rawId(), 1));
    }
  }

  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  e.getByToken(trackerEvent_token_, measurementTrackerEvent);

  const auto tkgeom = &es.getData(tkGeomToken_);
  const auto& stripcpe = es.getData(stripCPEToken_);
  const auto& stripQuality = es.getData(stripQualityToken_);
  const auto& magField = es.getData(magFieldToken_);
  const auto& measTracker = es.getData(measTrackerToken_);
  const auto& chi2Estimator = es.getData(chi2EstimatorToken_);
  const auto& prop = es.getData(propagatorToken_);

  // Tracking
  LogDebug("SiStripHitEfficiencyWorker") << "number ckf tracks found = " << tracksCKF->size();

  h_nTracks->Fill(tracksCKF->size());
  h_nTracksVsPU->Fill(PU, tracksCKF->size());

  // bin 0: one entry for each event
  calibData_.EventStats->Fill(0., 0., 1);
  // bin 1: one entry for each track
  calibData_.EventStats->Fill(1., 0., tracksCKF->size());

  if (!tracksCKF->empty()) {
    if (cutOnTracks_ && (tracksCKF->size() >= trackMultiplicityCut_))
      return;
    if (cutOnTracks_)
      LogDebug("SiStripHitEfficiencyWorker")
          << "starting checking good event with < " << trackMultiplicityCut_ << " tracks";

    // actually should do a loop over all the tracks in the event here

    // Looping over traj-track associations to be able to get traj & track informations
    for (const auto& trajTrack : *trajTrackAssociationHandle) {
      // for each track, fill some variables such as number of hits and momentum

      const bool highPurity = trajTrack.val->quality(reco::TrackBase::TrackQuality::highPurity);
      auto TMeas = trajTrack.key->measurements();
      totalNbHits += int(TMeas.size());

      /*
      const bool hasMissingHits = std::any_of(std::begin(TMeas), std::end(TMeas), [](const auto& tm) {
        return tm.recHit()->getType() == TrackingRecHit::Type::missing;
      });
      */

      // Check whether the trajectory has some missing hits
      bool hasMissingHits{false};
      int previous_layer{999};
      std::vector<unsigned int> missedLayers;

      for (const auto& itm : TMeas) {
        auto theHit = itm.recHit();
        unsigned int iidd = theHit->geographicalId().rawId();
        int layer = ::checkLayer(iidd, tTopo);
        int missedLayer = layer + 1;
        int diffPreviousLayer = (layer - previous_layer);
        if (doMissingHitsRecovery_) {
          //Layers from TIB + TOB
          if (diffPreviousLayer == -2 && missedLayer > k_LayersStart && missedLayer < k_LayersAtTOBEnd) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }
          //Layers from TID
          else if (diffPreviousLayer == -2 && (missedLayer > k_LayersAtTOBEnd + 1 && missedLayer <= k_LayersAtTIDEnd)) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }
          //Layers from TEC
          else if (diffPreviousLayer == -2 && missedLayer > k_LayersAtTIDEnd && missedLayer <= k_LayersAtTECEnd) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }

          //##### TID Layer 11 in transition TID -> TIB : layer is in TIB, previous layer  = 12
          if ((layer > k_LayersStart && layer <= k_LayersAtTIBEnd) && (previous_layer == 12)) {
            missHitPerLayer[11] += 1;
            hasMissingHits = true;
          }

          //##### TEC Layer 14 in transition TEC -> TOB : layer is in TOB, previous layer =  15
          if ((layer > k_LayersAtTIBEnd && layer <= k_LayersAtTOBEnd) && (previous_layer == 15)) {
            missHitPerLayer[14] += 1;
            hasMissingHits = true;
          }
        }
        if (theHit->getType() == TrackingRecHit::Type::missing)
          hasMissingHits = true;

        if (hasMissingHits)
          missedLayers.push_back(layer);
        previous_layer = layer;
      }

      // Loop on each measurement and take into consideration
      //--------------------------------------------------------
      unsigned int prev_TKlayers = 0;
      for (auto itm = TMeas.cbegin(); itm != TMeas.cend(); ++itm) {
        const auto theInHit = (*itm).recHit();

        //bin 2: one entry for each measurement
        calibData_.EventStats->Fill(2., 0., 1.);

        LogDebug("SiStripHitEfficiencyWorker") << "theInHit is valid = " << theInHit->isValid();

        unsigned int iidd = theInHit->geographicalId().rawId();

        unsigned int TKlayers = ::checkLayer(iidd, tTopo);

        // do not bother with pixel hits
        if (DetId(iidd).subdetId() < SiStripSubdetector::TIB)
          continue;

        LogDebug("SiStripHitEfficiencyWorker") << "TKlayer from trajectory: " << TKlayers << "  from module = " << iidd
                                               << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                                               << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2);

        // Test first and last points of the trajectory
        // the list of measurements starts from outer layers  !!! This could change -> should add a check
        bool isFirstMeas = (itm == (TMeas.end() - 1));
        bool isLastMeas = (itm == (TMeas.begin()));

        if (!useFirstMeas_ && isFirstMeas)
          continue;
        if (!useLastMeas_ && isLastMeas)
          continue;

        // In case of missing hit in the track, check whether to use the other hits or not.
        if (hasMissingHits && theInHit->getType() != TrackingRecHit::Type::missing &&
            !useAllHitsFromTracksWithMissingHits_)
          continue;

        // If Trajectory measurement from TOB 6 or TEC 9, skip it because it's always valid they are filled later
        if (TKlayers == bounds::k_LayersAtTOBEnd || TKlayers == bounds::k_LayersAtTECEnd) {
          LogDebug("SiStripHitEfficiencyWorker") << "skipping original TM for TOB 6 or TEC 9";
          continue;
        }

        std::vector<TrajectoryAtInvalidHit> TMs;

        // Make AnalyticalPropagat // TODO where to save these?or to use in TAVH constructor
        AnalyticalPropagator propagator(&magField, anyDirection);

        // for double sided layers check both sensors--if no hit was found on either sensor surface,
        // the trajectory measurements only have one invalid hit entry on the matched surface
        // so get the TrajectoryAtInvalidHit for both surfaces and include them in the study
        if (::isDoubleSided(iidd, tTopo) && ((iidd & 0x3) == 0)) {
          // do hit eff check twice--once for each sensor
          //add a TM for each surface
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 1);
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 2);
        } else if (::isDoubleSided(iidd, tTopo) && (!::check2DPartner(iidd, TMeas))) {
          // if only one hit was found the trajectory measurement is on that sensor surface, and the other surface from
          // the matched layer should be added to the study as well
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 1);
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator, 2);
          LogDebug("SiStripHitEfficiencyWorker") << " found a hit with a missing partner";
        } else {
          //only add one TM for the single surface and the other will be added in the next iteration
          TMs.emplace_back(*itm, tTopo, tkgeom, propagator);
        }

        bool missingHitAdded{false};
        std::vector<TrajectoryMeasurement> tmpTmeas;
        unsigned int misLayer = TKlayers + 1;
        //Use bool doMissingHitsRecovery to add possible missing hits based on actual/previous hit
        if (doMissingHitsRecovery_) {
          if (int(TKlayers) - int(prev_TKlayers) == -2) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{measTracker, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, prop, chi2Estimator);

            if (misLayer > k_LayersAtTIDEnd && misLayer < k_LayersAtTECEnd) {  //TEC
              std::vector<ForwardDetLayer const*> negTECLayers = measTracker.geometricSearchTracker()->negTecLayers();
              std::vector<ForwardDetLayer const*> posTECLayers = measTracker.geometricSearchTracker()->posTecLayers();
              const DetLayer* tecLayerneg = negTECLayers[misLayer - k_LayersAtTIDEnd - 1];
              const DetLayer* tecLayerpos = posTECLayers[misLayer - k_LayersAtTIDEnd - 1];
              if (tTopo->tecSide(iidd) == 1) {
                tmpTmeas = layerMeasurements.measurements(*tecLayerneg, tsos, prop, chi2Estimator);
              } else if (tTopo->tecSide(iidd) == 2) {
                tmpTmeas = layerMeasurements.measurements(*tecLayerpos, tsos, prop, chi2Estimator);
              }
            }

            else if (misLayer == (k_LayersAtTIDEnd - 1) ||
                     misLayer == k_LayersAtTIDEnd) {  // This is for TID layers 12 and 13
              std::vector<ForwardDetLayer const*> negTIDLayers = measTracker.geometricSearchTracker()->negTidLayers();
              std::vector<ForwardDetLayer const*> posTIDLayers = measTracker.geometricSearchTracker()->posTidLayers();
              const DetLayer* tidLayerneg = negTIDLayers[misLayer - k_LayersAtTOBEnd - 1];
              const DetLayer* tidLayerpos = posTIDLayers[misLayer - k_LayersAtTOBEnd - 1];

              if (tTopo->tidSide(iidd) == 1) {
                tmpTmeas = layerMeasurements.measurements(*tidLayerneg, tsos, prop, chi2Estimator);
              } else if (tTopo->tidSide(iidd) == 2) {
                tmpTmeas = layerMeasurements.measurements(*tidLayerpos, tsos, prop, chi2Estimator);
              }
            }

            if (misLayer > k_LayersStart && misLayer < k_LayersAtTOBEnd) {  // Barrel

              std::vector<BarrelDetLayer const*> barrelTIBLayers = measTracker.geometricSearchTracker()->tibLayers();
              std::vector<BarrelDetLayer const*> barrelTOBLayers = measTracker.geometricSearchTracker()->tobLayers();

              if (misLayer > k_LayersStart && misLayer <= k_LayersAtTIBEnd) {
                const DetLayer* tibLayer = barrelTIBLayers[misLayer - k_LayersStart - 1];
                tmpTmeas = layerMeasurements.measurements(*tibLayer, tsos, prop, chi2Estimator);
              } else if (misLayer > k_LayersAtTIBEnd && misLayer < k_LayersAtTOBEnd) {
                const DetLayer* tobLayer = barrelTOBLayers[misLayer - k_LayersAtTIBEnd - 1];
                tmpTmeas = layerMeasurements.measurements(*tobLayer, tsos, prop, chi2Estimator);
              }
            }
          }
          if ((int(TKlayers) > k_LayersStart && int(TKlayers) <= k_LayersAtTIBEnd) && int(prev_TKlayers) == 12) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{measTracker, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, prop, chi2Estimator);
            std::vector<ForwardDetLayer const*> negTIDLayers = measTracker.geometricSearchTracker()->negTidLayers();
            std::vector<ForwardDetLayer const*> posTIDLayers = measTracker.geometricSearchTracker()->posTidLayers();

            const DetLayer* tidLayerneg = negTIDLayers[k_LayersStart];
            const DetLayer* tidLayerpos = posTIDLayers[k_LayersStart];
            if (tTopo->tidSide(iidd) == 1) {
              tmpTmeas = layerMeasurements.measurements(*tidLayerneg, tsos, prop, chi2Estimator);
            } else if (tTopo->tidSide(iidd) == 2) {
              tmpTmeas = layerMeasurements.measurements(*tidLayerpos, tsos, prop, chi2Estimator);
            }
          }

          if ((int(TKlayers) > k_LayersAtTIBEnd && int(TKlayers) <= k_LayersAtTOBEnd) && int(prev_TKlayers) == 15) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{measTracker, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, prop, chi2Estimator);

            std::vector<ForwardDetLayer const*> negTECLayers = measTracker.geometricSearchTracker()->negTecLayers();
            std::vector<ForwardDetLayer const*> posTECLayers = measTracker.geometricSearchTracker()->posTecLayers();

            const DetLayer* tecLayerneg = negTECLayers[k_LayersStart];
            const DetLayer* tecLayerpos = posTECLayers[k_LayersStart];
            if (tTopo->tecSide(iidd) == 1) {
              tmpTmeas = layerMeasurements.measurements(*tecLayerneg, tsos, prop, chi2Estimator);
            } else if (tTopo->tecSide(iidd) == 2) {
              tmpTmeas = layerMeasurements.measurements(*tecLayerpos, tsos, prop, chi2Estimator);
            }
          }

          if (!tmpTmeas.empty()) {
            TrajectoryMeasurement TM_tmp(tmpTmeas.back());
            unsigned int iidd_tmp = TM_tmp.recHit()->geographicalId().rawId();
            if (iidd_tmp != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << " hit actually being added to TM vector";
              if ((!useAllHitsFromTracksWithMissingHits_ || (!useFirstMeas_ && isFirstMeas)))
                TMs.clear();
              if (::isDoubleSided(iidd_tmp, tTopo)) {
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator, 1));
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator, 2));
              } else
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator));
              missingHitAdded = true;
              hitRecoveryCounters[misLayer] += 1;
            }
          }
        }

        prev_TKlayers = TKlayers;
        if (!useFirstMeas_ && isFirstMeas && !missingHitAdded)
          continue;
        if (!useLastMeas_ && isLastMeas)
          continue;
        bool hitsWithBias = false;
        for (auto ilayer : missedLayers) {
          if (ilayer < TKlayers)
            hitsWithBias = true;
        }
        if (hasMissingHits && theInHit->getType() != TrackingRecHit::Type::missing && !missingHitAdded &&
            hitsWithBias && !useAllHitsFromTracksWithMissingHits_) {
          continue;
        }

        //////////////////////////////////////////////
        //Now check for tracks at TOB6 and TEC9

        // to make sure we only propagate on the last TOB5 hit check the next entry isn't also in TOB5
        // to avoid bias, make sure the TOB5 hit is valid (an invalid hit on TOB5 could only exist with a valid hit on TOB6)
        const auto nextId = (itm + 1 != TMeas.end()) ? (itm + 1)->recHit()->geographicalId() : DetId{};  // null if last

        if (TKlayers == 9 && theInHit->isValid() && !((!nextId.null()) && (::checkLayer(nextId.rawId(), tTopo) == 9))) {
          //	  if ( TKlayers==9 && itm==TMeas.rbegin()) {
          //	  if ( TKlayers==9 && (itm==TMeas.back()) ) {	  // to check for only the last entry in the trajectory for propagation
          const DetLayer* tob6 = measTracker.geometricSearchTracker()->tobLayers().back();
          const LayerMeasurements theLayerMeasurements{measTracker, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTOB5 = itm->updatedState();
          const auto tmp = theLayerMeasurements.measurements(*tob6, tsosTOB5, prop, chi2Estimator);

          if (!tmp.empty()) {
            LogDebug("SiStripHitEfficiencyWorker") << "size of TM from propagation = " << tmp.size();

            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            const auto& tob6TM = tmp.back();
            const auto& tob6Hit = tob6TM.recHit();
            if (tob6Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiencyWorker") << "tob6 hit actually being added to TM vector";
              TMs.emplace_back(tob6TM, tTopo, tkgeom, propagator);
            }
          }
        }

        // same for TEC8
        if (TKlayers == 21 && theInHit->isValid() &&
            !((!nextId.null()) && (::checkLayer(nextId.rawId(), tTopo) == 21))) {
          const DetLayer* tec9pos = measTracker.geometricSearchTracker()->posTecLayers().back();
          const DetLayer* tec9neg = measTracker.geometricSearchTracker()->negTecLayers().back();

          const LayerMeasurements theLayerMeasurements{measTracker, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTEC9 = itm->updatedState();

          // check if track on positive or negative z
          if (!(iidd == SiStripSubdetector::TEC))
            LogDebug("SiStripHitEfficiencyWorker") << "there is a problem with TEC 9 extrapolation";

          //LogDebug("SiStripHitEfficiencyWorker") << " tec9 id = " << iidd << " and side = " << tTopo->tecSide(iidd) ;
          std::vector<TrajectoryMeasurement> tmp;
          if (tTopo->tecSide(iidd) == 1) {
            tmp = theLayerMeasurements.measurements(*tec9neg, tsosTEC9, prop, chi2Estimator);
            //LogDebug("SiStripHitEfficiencyWorker") << "on negative side" ;
          }
          if (tTopo->tecSide(iidd) == 2) {
            tmp = theLayerMeasurements.measurements(*tec9pos, tsosTEC9, prop, chi2Estimator);
            //LogDebug("SiStripHitEfficiencyWorker") << "on positive side" ;
          }

          if (!tmp.empty()) {
            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            const auto& tec9TM = tmp.back();
            const auto& tec9Hit = tec9TM.recHit();

            const unsigned int tec9id = tec9Hit->geographicalId().rawId();
            LogDebug("SiStripHitEfficiencyWorker")
                << "tec9id = " << tec9id << " is Double sided = " << ::isDoubleSided(tec9id, tTopo)
                << "  and 0x3 = " << (tec9id & 0x3);

            if (tec9Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiencyWorker") << "tec9 hit actually being added to TM vector";
              // in tec the hit can be single or doubled sided. whenever the invalid hit at the end of vector of TMs is
              // double sided it is always on the matched surface, so we need to split it into the true sensor surfaces
              if (::isDoubleSided(tec9id, tTopo)) {
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator, 1);
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator, 2);
              } else
                TMs.emplace_back(tec9TM, tTopo, tkgeom, propagator);
            }
          }  //else LogDebug("SiStripHitEfficiencyWorker") << "tec9 tmp empty" ;
        }

        for (const auto& tm : TMs) {
          fillForTraj(tm,
                      tTopo,
                      tkgeom,
                      stripcpe,
                      stripQuality,
                      *fedErrorIds,
                      commonModeDigis,
                      *theClusters,
                      e.bunchCrossing(),
                      instLumi,
                      PU,
                      highPurity);
        }
        LogDebug("SiStripHitEfficiencyWorker") << "After looping over TrajAtValidHit list";
      }
      LogDebug("SiStripHitEfficiencyWorker") << "end TMeasurement loop";
    }
    LogDebug("SiStripHitEfficiencyWorker") << "end of trajectories loop";
  }
}

void SiStripHitEfficiencyWorker::fillForTraj(const TrajectoryAtInvalidHit& tm,
                                             const TrackerTopology* tTopo,
                                             const TrackerGeometry* tkgeom,
                                             const StripClusterParameterEstimator& stripCPE,
                                             const SiStripQuality& stripQuality,
                                             const DetIdCollection& fedErrorIds,
                                             const edm::Handle<edm::DetSetVector<SiStripRawDigi>>& commonModeDigis,
                                             const edmNew::DetSetVector<SiStripCluster>& theClusters,
                                             int bunchCrossing,
                                             float instLumi,
                                             float PU,
                                             bool highPurity) {
  // --> Get trajectory from combinatedStat& e
  const auto iidd = tm.monodet_id();
  LogDebug("SiStripHitEfficiencyWorker") << "setting iidd = " << iidd << " before checking efficiency and ";

  const auto xloc = tm.localX();
  const auto yloc = tm.localY();

  const auto xErr = tm.localErrorX();
  const auto yErr = tm.localErrorY();

  int TrajStrip = -1;

  // reget layer from iidd here, to account for TOB 6 and TEC 9 TKlayers being off
  const auto TKlayers = ::checkLayer(iidd, tTopo);

  const bool withinAcceptance =
      tm.withinAcceptance() && (!::isInBondingExclusionZone(iidd, TKlayers, yloc, yErr, tTopo));

  if (  // (TKlayers > 0) && // FIXME confirm this
      ((layers_ == TKlayers) ||
       (layers_ == bounds::k_LayersStart))) {  // Look at the layer not used to reconstruct the track
    LogDebug("SiStripHitEfficiencyWorker") << "Looking at layer under study";
    unsigned int ModIsBad = 2;
    unsigned int SiStripQualBad = 0;
    float commonMode = -100;

    // RPhi RecHit Efficiency

    if (!theClusters.empty()) {
      LogDebug("SiStripHitEfficiencyWorker") << "Checking clusters with size = " << theClusters.size();
      std::vector<::ClusterInfo> VCluster_info;  //fill with X residual, X residual pull, local X
      const auto idsv = theClusters.find(iidd);
      if (idsv != theClusters.end()) {
        //if (DEBUG_)      LogDebug("SiStripHitEfficiencyWorker") << "the ID from the dsv = " << dsv.id() ;
        LogDebug("SiStripHitEfficiencyWorker")
            << "found  (ClusterId == iidd) with ClusterId = " << idsv->id() << " and iidd = " << iidd;
        const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(iidd)));
        const StripTopology& Topo = stripdet->specificTopology();

        float hbedge = 0.0;
        float htedge = 0.0;
        float hapoth = 0.0;
        float uylfac = 0.0;
        float uxlden = 0.0;
        if (TKlayers > bounds::k_LayersAtTOBEnd) {
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
          if (TKlayers > bounds::k_LayersAtTOBEnd) {
            const float TrajLocXMid = xloc / (1 + (htedge - hbedge) * yloc / (htedge + hbedge) /
                                                      hapoth);  // radialy extrapolated x loc position at middle
            TrajStrip = TrajLocXMid / pitch + nstrips / 2.0;
          }
          //LogDebug("SiStripHitEfficiency")<<" Layer "<<TKlayers<<" TrajStrip: "<<nstrips<<" "<<pitch<<" "<<TrajStrip;;
        }

        for (const auto& clus : *idsv) {
          StripClusterParameterEstimator::LocalValues parameters = stripCPE.localParameters(clus, *stripdet);
          float res = (parameters.first.x() - xloc);
          float sigma = ::checkConsistency(parameters, xloc, xErr);
          // The consistency is probably more accurately measured with the Chi2MeasurementEstimator. To use it
          // you need a TransientTrackingRecHit instead of the cluster
          //theEstimator=       new Chi2MeasurementEstimator(30);
          //const Chi2MeasurementEstimator *theEstimator(100);
          //theEstimator->estimate(tm.tsos(), TransientTrackingRecHit);

          if (TKlayers > bounds::k_LayersAtTOBEnd) {
            res = parameters.first.x() - xloc / uxlden;  // radialy extrapolated x loc position at middle
            sigma = abs(res) / sqrt(parameters.second.xx() + xErr * xErr / uxlden / uxlden +
                                    yErr * yErr * xloc * xloc * uylfac * uylfac / uxlden / uxlden / uxlden / uxlden);
          }

          VCluster_info.emplace_back(res, sigma, parameters.first.x());

          LogDebug("SiStripHitEfficiencyWorker") << "Have ID match. residual = " << res << "  res sigma = " << sigma;
          //LogDebug("SiStripHitEfficiencyWorker")
          //    << "trajectory measurement compatability estimate = " << (*itm).estimate() ;
          LogDebug("SiStripHitEfficiencyWorker")
              << "hit position = " << parameters.first.x() << "  hit error = " << sqrt(parameters.second.xx())
              << "  trajectory position = " << xloc << "  traj error = " << xErr;
        }
      }
      ::ClusterInfo finalCluster{1000.0, 1000.0, 0.0};
      if (!VCluster_info.empty()) {
        LogDebug("SiStripHitEfficiencyWorker") << "found clusters > 0";
        if (VCluster_info.size() > 1) {
          //get the smallest one
          for (const auto& res : VCluster_info) {
            if (std::abs(res.xResidualPull) < std::abs(finalCluster.xResidualPull)) {
              finalCluster = res;
            }
            LogDebug("SiStripHitEfficiencyWorker")
                << "iresidual = " << res.xResidual << "  isigma = " << res.xResidualPull
                << "  and FinalRes = " << finalCluster.xResidual;
          }
        } else {
          finalCluster = VCluster_info[0];
        }
        VCluster_info.clear();
      }

      LogDebug("SiStripHitEfficiencyWorker") << "Final residual in X = " << finalCluster.xResidual << "+-"
                                             << (finalCluster.xResidual / finalCluster.xResidualPull);
      LogDebug("SiStripHitEfficiencyWorker")
          << "Checking location of trajectory: abs(yloc) = " << abs(yloc) << "  abs(xloc) = " << abs(xloc);

      //
      // fill ntuple varibles

      //if ( stripQuality->IsModuleBad(iidd) )
      if (stripQuality.getBadApvs(iidd) != 0) {
        SiStripQualBad = 1;
        LogDebug("SiStripHitEfficiencyWorker") << "strip is bad from SiStripQuality";
      } else {
        SiStripQualBad = 0;
        LogDebug("SiStripHitEfficiencyWorker") << "strip is good from SiStripQuality";
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
            if ((unsigned)TrajStrip / sistrip::STRIPS_PER_APV < digiframe->data.size())
              commonMode = digiframe->data.at(TrajStrip / sistrip::STRIPS_PER_APV).adc();
        }

      LogDebug("SiStripHitEfficiencyWorker") << "before check good";

      if (finalCluster.xResidualPull < 999.0) {  //could make requirement on track/hit consistency, but for
        //now take anything with a hit on the module
        LogDebug("SiStripHitEfficiencyWorker")
            << "hit being counted as good " << finalCluster.xResidual << " FinalRecHit " << iidd << "   TKlayers  "
            << TKlayers << " xloc " << xloc << " yloc  " << yloc << " module " << iidd
            << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1) << "/"
            << ((iidd & 0x3) == 2);
        ModIsBad = 0;
      } else {
        LogDebug("SiStripHitEfficiencyWorker")
            << "hit being counted as bad   ######### Invalid RPhi FinalResX " << finalCluster.xResidual
            << " FinalRecHit " << iidd << "   TKlayers  " << TKlayers << " xloc " << xloc << " yloc  " << yloc
            << " module " << iidd << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1)
            << "/" << ((iidd & 0x3) == 2);
        ModIsBad = 1;
        LogDebug("SiStripHitEfficiencyWorker")
            << " RPhi Error " << sqrt(xErr * xErr + yErr * yErr) << " ErrorX " << xErr << " yErr " << yErr;
      }

      LogDebug("SiStripHitEfficiencyWorker")
          << "To avoid them staying unused: ModIsBad=" << ModIsBad << ", SiStripQualBad=" << SiStripQualBad
          << ", commonMode=" << commonMode << ", highPurity=" << highPurity
          << ", withinAcceptance=" << withinAcceptance;

      unsigned int layer = TKlayers;
      if (showRings_ && layer > bounds::k_LayersAtTOBEnd) {  // use rings instead of wheels
        if (layer <= bounds::k_LayersAtTIDEnd) {             // TID
          layer = bounds::k_LayersAtTOBEnd +
                  tTopo->tidRing(iidd);  // ((iidd >> 9) & 0x3);  // 3 disks and also 3 rings -> use the same container
        } else {                         // TEC
          layer = bounds::k_LayersAtTIDEnd + tTopo->tecRing(iidd);  // ((iidd >> 5) & 0x7);
        }
      }
      unsigned int layerWithSide = layer;
      if (layer > bounds::k_LayersAtTOBEnd && layer <= bounds::k_LayersAtTIDEnd) {
        const auto side = tTopo->tidSide(iidd);  //(iidd >> 13) & 0x3;  // TID
        if (side == 2)
          layerWithSide = layer + 3;
      } else if (layer > bounds::k_LayersAtTIDEnd) {
        const auto side = tTopo->tecSide(iidd);  // (iidd >> 18) & 0x3;  // TEC
        if (side == 1) {
          layerWithSide = layer + 3;
        } else if (side == 2) {
          layerWithSide = layer + 3 + (showRings_ ? 7 : 9);
        }
      }

      if ((bunchX_ > 0 && bunchX_ != bunchCrossing) || (!withinAcceptance) ||
          (useOnlyHighPurityTracks_ && !highPurity) ||
          (!showTOB6TEC9_ && (TKlayers == bounds::k_LayersAtTOBEnd || TKlayers == bounds::k_LayersAtTECEnd)) ||
          (badModules_.end() != badModules_.find(iidd)))
        return;

      const bool badquality = (SiStripQualBad == 1);

      //Now that we have a good event, we need to look at if we expected it or not, and the location
      //if we didn't
      //Fill the missing hit information first
      bool badflag = false;  // true for hits that are expected but not found
      if (resXSig_ < 0) {
        if (ModIsBad == 1)
          badflag = true;  // isBad set to false in the tree when resxsig<999.0
      } else {
        if (ModIsBad == 1 || finalCluster.xResidualPull > resXSig_)
          badflag = true;
      }

      // Conversion of positions in strip unit
      int nstrips = -9;
      float Pitch = -9.0;
      const StripGeomDetUnit* stripdet = nullptr;
      if (finalCluster.xResidualPull ==
          1000.0) {      // special treatment, no GeomDetUnit associated in some cases when no cluster found
        Pitch = 0.0205;  // maximum
        nstrips = 768;   // maximum
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
      if (stripdet && layer > bounds::k_LayersAtTOBEnd) {
        const auto& trapezoidalBounds = dynamic_cast<const TrapezoidalPlaneBounds&>(stripdet->surface().bounds());
        std::array<const float, 4> const& parameters = trapezoidalBounds.parameters();
        const float hbedge = parameters[0];
        const float htedge = parameters[1];
        const float hapoth = parameters[3];
        const float TrajLocXMid = xloc / (1 + (htedge - hbedge) * yloc / (htedge + hbedge) /
                                                  hapoth);  // radialy extrapolated x loc position at middle
        stripTrajMid = TrajLocXMid / Pitch + nstrips / 2.0;
      }

      if ((!badquality) && (layer < h_resolution.size())) {
        LogDebug("SiStripHitEfficiencyWorker")
            << "layer " << layer << " vector index " << layer - 1 << " before filling h_resolution" << std::endl;
        h_resolution[layer - 1]->Fill(finalCluster.xResidualPull != 1000.0 ? stripTrajMid - stripCluster : 1000);
      }

      // New matching methods
      if (clusterMatchingMethod_ >= 1) {
        badflag = false;
        if (finalCluster.xResidualPull == 1000.0) {
          LogDebug("SiStripHitEfficiencyWorker") << "Marking bad for resxsig=1000";
          badflag = true;
        } else {
          if (clusterMatchingMethod_ == 2 || clusterMatchingMethod_ == 4) {
            // check the distance between cluster and trajectory position
            if (std::abs(stripCluster - stripTrajMid) > clusterTracjDist_) {
              LogDebug("SiStripHitEfficiencyWorker") << "Marking bad for cluster-to-traj distance";
              badflag = true;
            }
          }
          if (clusterMatchingMethod_ == 3 || clusterMatchingMethod_ == 4) {
            // cluster and traj have to be in the same APV (don't take edges into accounts)
            const int tapv = (int)stripTrajMid / sistrip::STRIPS_PER_APV;
            const int capv = (int)stripCluster / sistrip::STRIPS_PER_APV;
            float stripInAPV = stripTrajMid - tapv * sistrip::STRIPS_PER_APV;
            if (stripInAPV < stripsApvEdge_ || stripInAPV > sistrip::STRIPS_PER_APV - stripsApvEdge_) {
              LogDebug("SiStripHitEfficiencyWorker") << "Too close to the edge: " << stripInAPV;
              return;
            }
            if (tapv != capv) {
              LogDebug("SiStripHitEfficiencyWorker") << "Marking bad for tapv!=capv";
              badflag = true;
            }
          }
        }
      }
      if (!badquality) {
        LogDebug("SiStripHitEfficiencyWorker")
            << "Filling measurement for " << iidd << " in layer " << layer << " histograms with bx=" << bunchCrossing
            << ", lumi=" << instLumi << ", PU=" << PU << "; bad flag=" << badflag;

        // hot/cold maps of hits that are expected but not found
        if (badflag) {
          if (layer > bounds::k_LayersStart && layer <= bounds::k_LayersAtTIBEnd) {
            //We are in the TIB
            float phi = ::calcPhi(tm.globalX(), tm.globalY());
            h_hotcold[layer - 1]->Fill(360. - phi, tm.globalZ(), 1.);
          } else if (layer > bounds::k_LayersAtTIBEnd && layer <= bounds::k_LayersAtTOBEnd) {
            //We are in the TOB
            float phi = ::calcPhi(tm.globalX(), tm.globalY());
            h_hotcold[layer - 1]->Fill(360. - phi, tm.globalZ(), 1.);
          } else if (layer > bounds::k_LayersAtTOBEnd && layer <= bounds::k_LayersAtTIDEnd) {
            //We are in the TID
            //There are 2 different maps here
            int side = tTopo->tidSide(iidd);
            if (side == 1)
              h_hotcold[(layer - 1) + (layer - 11)]->Fill(-tm.globalY(), tm.globalX(), 1.);
            else if (side == 2)
              h_hotcold[(layer - 1) + (layer - 10)]->Fill(-tm.globalY(), tm.globalX(), 1.);
          } else if (layer > bounds::k_LayersAtTIDEnd) {
            //We are in the TEC
            //There are 2 different maps here
            int side = tTopo->tecSide(iidd);
            if (side == 1)
              h_hotcold[(layer + 2) + (layer - 14)]->Fill(-tm.globalY(), tm.globalX(), 1.);
            else if (side == 2)
              h_hotcold[(layer + 2) + (layer - 13)]->Fill(-tm.globalY(), tm.globalX(), 1.);
          }
        }

        LogDebug("SiStripHitEfficiencyWorker")
            << "layer " << layer << " vector index " << layer - 1 << " before filling h_layer_vsSmthg" << std::endl;
        h_layer_vsBx[layer - 1].fill(bunchCrossing, !badflag);
        if (addLumi_) {
          h_layer_vsLumi[layer - 1].fill(instLumi, !badflag);
          h_layer_vsPU[layer - 1].fill(PU, !badflag);
        }
        if (addCommonMode_) {
          h_layer_vsCM[layer - 1].fill(commonMode, !badflag);
        }
        h_goodLayer.fill(layerWithSide, !badflag);
      }
      // efficiency without bad modules excluded
      h_allLayer.fill(layerWithSide, !badflag);

      // efficiency without bad modules excluded
      if (TKlayers) {
        h_module.fill(iidd, !badflag);
        assert(h_module.check(iidd));
      }

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
      LogDebug("SiStripHitEfficiencyWorker") << "after good location check";
    }
    LogDebug("SiStripHitEfficiencyWorker") << "after list of clusters";
  }
  LogDebug("SiStripHitEfficiencyWorker") << "After layers=TKLayers if with TKlayers=" << TKlayers
                                         << ", layers=" << layers_;
}

void SiStripHitEfficiencyWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("dqmDir", "AlCaReco/SiStripHitEfficiency");
  desc.add<bool>("UseOnlyHighPurityTracks", true);
  desc.add<bool>("cutOnTracks", false);
  desc.add<bool>("doMissingHitsRecovery", false);
  desc.add<bool>("useAllHitsFromTracksWithMissingHits", false);
  desc.add<bool>("useFirstMeas", false);
  desc.add<bool>("useLastMeas", false);
  desc.add<double>("ClusterTrajDist", 64.0);
  desc.add<double>("ResXSig", -1);
  desc.add<double>("StripsApvEdge", 10.0);
  desc.add<edm::InputTag>("combinatorialTracks", edm::InputTag{"generalTracks"});
  desc.add<edm::InputTag>("commonMode", edm::InputTag{"siStripDigis", "CommonMode"});
  desc.add<edm::InputTag>("lumiScalers", edm::InputTag{"scalersRawToDigi"});
  desc.add<edm::InputTag>("metadata", edm::InputTag{"onlineMetaDataDigis"});
  desc.add<edm::InputTag>("siStripClusters", edm::InputTag{"siStripClusters"});
  desc.add<edm::InputTag>("siStripDigis", edm::InputTag{"siStripDigis"});
  desc.add<edm::InputTag>("trackerEvent", edm::InputTag{"MeasurementTrackerEvent"});
  desc.add<edm::InputTag>("trajectories", edm::InputTag{"generalTracks"});
  desc.add<int>("ClusterMatchingMethod", 0);
  desc.add<int>("Layer", 0);
  desc.add<unsigned int>("trackMultiplicity", 100);
  desc.addUntracked<bool>("Debug", false);
  desc.addUntracked<bool>("ShowRings", false);
  desc.addUntracked<bool>("ShowTOB6TEC9", false);
  desc.addUntracked<bool>("addCommonMode", false);
  desc.addUntracked<bool>("addLumi", true);
  desc.addUntracked<int>("BunchCrossing", 0);
  desc.addUntracked<std::string>("BadModulesFile", "");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyWorker);
