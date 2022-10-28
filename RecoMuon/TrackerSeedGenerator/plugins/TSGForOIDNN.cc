/**
  \class    TSGForOIDNN
  \brief    Create L3MuonTrajectorySeeds from L2 Muons in an outside-in manner
  \author   Dmitry Kondratyev, Arnab Purohit, Jan-Frederik Schulte (Purdue University, West Lafayette, USA)
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <memory>
namespace pt = boost::property_tree;

class TSGForOIDNN : public edm::global::EDProducer<> {
public:
  explicit TSGForOIDNN(const edm::ParameterSet& iConfig);
  ~TSGForOIDNN() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  /// Labels for input collections
  const edm::EDGetTokenT<reco::TrackCollection> src_;
  /// Tokens for ESHandle
  const edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> t_estimatorH_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> t_magfieldH_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_propagatorAlongH_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_propagatorOppositeH_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> t_tmpTkGeometryH_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> t_geometryH_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> t_navSchool_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_SHPOpposite_;
  const std::unique_ptr<TrajectoryStateUpdator> updator_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;
  const std::string theCategory_;

  /// Maximum number of seeds for each L2
  const unsigned int maxSeeds_;
  /// Maximum number of hitbased seeds for each L2
  const unsigned int maxHitSeeds_;
  /// Maximum number of hitless seeds for each L2
  const unsigned int maxHitlessSeeds_;
  /// How many layers to try
  const unsigned int numOfLayersToTry_;
  /// How many hits to try per layer
  const unsigned int numOfHitsToTry_;
  /// Rescale L2 parameter uncertainties (fixed error vs pT, eta)
  const double fixedErrorRescalingForHitless_;
  /// Estimator used to find dets and TrajectoryMeasurements
  const std::string estimatorName_;
  /// Minimum eta value to activate searching in the TEC
  const double minEtaForTEC_;
  /// Maximum eta value to activate searching in the TOB
  const double maxEtaForTOB_;

  /// IP refers to TSOS at interaction point,
  /// MuS refers to TSOS at muon system
  const unsigned int maxHitlessSeedsIP_;
  const unsigned int maxHitlessSeedsMuS_;
  const unsigned int maxHitDoubletSeeds_;

  /// Get number of seeds to use from DNN output instead of "max..Seeds" parameters
  const bool getStrategyFromDNN_;
  /// Whether to use DNN regressor (if false, will use classifier)
  const bool useRegressor_;

  /// Settings for classifier
  std::string dnnModelPath_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_;
  tensorflow::Session* tf_session_;

  /// Settings for regressor
  std::string dnnModelPath_HB_;
  std::string dnnModelPath_HLIP_;
  std::string dnnModelPath_HLMuS_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_HB_;
  tensorflow::Session* tf_session_HB_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_HLIP_;
  tensorflow::Session* tf_session_HLIP_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_HLMuS_;
  tensorflow::Session* tf_session_HLMuS_;

  /// DNN metadata
  const std::string dnnMetadataPath_;
  pt::ptree metadata_;

  /// Create seeds without hits on a given layer (TOB or TEC)
  void makeSeedsWithoutHits(const GeometricSearchDet& layer,
                            const TrajectoryStateOnSurface& tsos,
                            const Propagator& propagatorAlong,
                            const Chi2MeasurementEstimatorBase& estimator,
                            double errorSF,
                            unsigned int& hitlessSeedsMade,
                            unsigned int& numSeedsMade,
                            std::vector<TrajectorySeed>& out) const;

  /// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
  void makeSeedsFromHits(const GeometricSearchDet& layer,
                         const TrajectoryStateOnSurface& tsos,
                         const Propagator& propagatorAlong,
                         const Chi2MeasurementEstimatorBase& estimator,
                         const MeasurementTrackerEvent& measurementTracker,
                         unsigned int& hitSeedsMade,
                         unsigned int& numSeedsMade,
                         const unsigned int& maxHitSeeds,
                         unsigned int& layerCount,
                         std::vector<TrajectorySeed>& out) const;

  /// Similar to makeSeedsFromHits, but seed is created only if there are compatible hits on two adjacent layers
  void makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                const TrajectoryStateOnSurface& tsos,
                                const Propagator& propagatorAlong,
                                const Chi2MeasurementEstimatorBase& estimator,
                                const MeasurementTrackerEvent& measurementTracker,
                                const NavigationSchool& navSchool,
                                unsigned int& hitDoubletSeedsMade,
                                unsigned int& numSeedsMade,
                                const unsigned int& maxHitDoubletSeeds,
                                unsigned int& layerCount,
                                std::vector<TrajectorySeed>& out) const;

  /// Update dictionary of inputs for DNN
  void updateFeatureMap(std::unordered_map<std::string, float>& the_map,
                        const reco::Track& l2,
                        const TrajectoryStateOnSurface& tsos_IP,
                        const TrajectoryStateOnSurface& tsos_MuS) const;

  /// Container for DNN outupts
  struct StrategyParameters {
    int nHBd, nHLIP, nHLMuS, sf;
  };

  /// Evaluate DNN classifier
  void evaluateClassifier(const std::unordered_map<std::string, float>& feature_map,
                          tensorflow::Session* session,
                          const pt::ptree& metadata,
                          StrategyParameters& out,
                          bool& dnnSuccess) const;

  /// Evaluate DNN regressor
  void evaluateRegressor(const std::unordered_map<std::string, float>& feature_map,
                         tensorflow::Session* session_HB,
                         const pt::ptree& metadata_HB,
                         tensorflow::Session* session_HLIP,
                         const pt::ptree& metadata_HLIP,
                         tensorflow::Session* session_HLMuS,
                         const pt::ptree& metadata_HLMuS,
                         StrategyParameters& out,
                         bool& dnnSuccess) const;
};

TSGForOIDNN::TSGForOIDNN(const edm::ParameterSet& iConfig)
    : src_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      t_estimatorH_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("estimator")))),
      t_magfieldH_(esConsumes()),
      t_propagatorAlongH_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_propagatorOppositeH_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_tmpTkGeometryH_(esConsumes()),
      t_geometryH_(esConsumes()),
      t_navSchool_(esConsumes(edm::ESInputTag("", "SimpleNavigationSchool"))),
      t_SHPOpposite_(esConsumes(edm::ESInputTag("", "hltESPSteppingHelixPropagatorOpposite"))),
      updator_(new KFUpdator()),
      measurementTrackerTag_(consumes(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      theCategory_(std::string("Muon|RecoMuon|TSGForOIDNN")),
      maxSeeds_(iConfig.getParameter<uint32_t>("maxSeeds")),
      maxHitSeeds_(iConfig.getParameter<uint32_t>("maxHitSeeds")),
      maxHitlessSeeds_(iConfig.getParameter<uint32_t>("maxHitlessSeeds")),
      numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
      numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
      fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
      minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
      maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
      maxHitlessSeedsIP_(iConfig.getParameter<uint32_t>("maxHitlessSeedsIP")),
      maxHitlessSeedsMuS_(iConfig.getParameter<uint32_t>("maxHitlessSeedsMuS")),
      maxHitDoubletSeeds_(iConfig.getParameter<uint32_t>("maxHitDoubletSeeds")),
      getStrategyFromDNN_(iConfig.getParameter<bool>("getStrategyFromDNN")),
      useRegressor_(iConfig.getParameter<bool>("useRegressor")),
      dnnMetadataPath_(iConfig.getParameter<std::string>("dnnMetadataPath")) {
  if (getStrategyFromDNN_) {
    edm::FileInPath dnnMetadataPath(dnnMetadataPath_);
    pt::read_json(dnnMetadataPath.fullPath(), metadata_);
    tensorflow::setLogging("3");

    if (useRegressor_) {
      // use regressor
      dnnModelPath_HB_ = metadata_.get<std::string>("HB.dnnmodel_path");
      edm::FileInPath dnnPath_HB(dnnModelPath_HB_);
      graphDef_HB_ = std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(dnnPath_HB.fullPath()));
      tf_session_HB_ = tensorflow::createSession(graphDef_HB_.get());

      dnnModelPath_HLIP_ = metadata_.get<std::string>("HLIP.dnnmodel_path");
      edm::FileInPath dnnPath_HLIP(dnnModelPath_HLIP_);
      graphDef_HLIP_ = std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(dnnPath_HLIP.fullPath()));
      tf_session_HLIP_ = tensorflow::createSession(graphDef_HLIP_.get());

      dnnModelPath_HLMuS_ = metadata_.get<std::string>("HLMuS.dnnmodel_path");
      edm::FileInPath dnnPath_HLMuS(dnnModelPath_HLMuS_);
      graphDef_HLMuS_ = std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(dnnPath_HLMuS.fullPath()));
      tf_session_HLMuS_ = tensorflow::createSession(graphDef_HLMuS_.get());
    } else {
      // use classifier (default)
      dnnModelPath_ = metadata_.get<std::string>("dnnmodel_path");
      edm::FileInPath dnnPath(dnnModelPath_);
      graphDef_ = std::unique_ptr<tensorflow::GraphDef>(tensorflow::loadGraphDef(dnnPath.fullPath()));
      tf_session_ = tensorflow::createSession(graphDef_.get());
    }
  }
  produces<std::vector<TrajectorySeed> >();
}

TSGForOIDNN::~TSGForOIDNN() {
  if (getStrategyFromDNN_) {
    if (useRegressor_) {
      tensorflow::closeSession(tf_session_HB_);
      tensorflow::closeSession(tf_session_HLIP_);
      tensorflow::closeSession(tf_session_HLMuS_);
    } else {
      tensorflow::closeSession(tf_session_);
    }
  }
}

//
// Produce seeds
//
void TSGForOIDNN::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& iEventSetup) const {
  // Initialize variables
  unsigned int numSeedsMade = 0;
  unsigned int layerCount = 0;
  unsigned int hitlessSeedsMadeIP = 0;
  unsigned int hitlessSeedsMadeMuS = 0;
  unsigned int hitSeedsMade = 0;
  unsigned int hitDoubletSeedsMade = 0;

  // Container for DNN inputs
  std::unordered_map<std::string, float> feature_map;

  // Container for DNN outputs
  StrategyParameters strPars;

  // Surface used to make a TSOS at the PCA to the beamline
  Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

  // Get setup objects
  const MagneticField& magfield = iEventSetup.getData(t_magfieldH_);
  const Chi2MeasurementEstimatorBase& estimator = iEventSetup.getData(t_estimatorH_);
  const Propagator& tmpPropagatorAlong = iEventSetup.getData(t_propagatorAlongH_);
  const Propagator& tmpPropagatorOpposite = iEventSetup.getData(t_propagatorOppositeH_);
  const TrackerGeometry& tmpTkGeometry = iEventSetup.getData(t_tmpTkGeometryH_);
  const GlobalTrackingGeometry& geometry = iEventSetup.getData(t_geometryH_);
  const NavigationSchool& navSchool = iEventSetup.getData(t_navSchool_);
  auto const& measurementTracker = iEvent.get(measurementTrackerTag_);

  // Read L2 track collection
  auto const& l2TrackCol = iEvent.get(src_);

  // The product
  std::unique_ptr<std::vector<TrajectorySeed> > result(new std::vector<TrajectorySeed>());

  // Get vector of Detector layers
  auto const* gsTracker = measurementTracker.geometricSearchTracker();
  std::vector<BarrelDetLayer const*> const& tob = gsTracker->tobLayers();
  std::vector<ForwardDetLayer const*> const& tecPositive =
      tmpTkGeometry.isThere(GeomDetEnumerators::P2OTEC) ? gsTracker->posTidLayers() : gsTracker->posTecLayers();
  std::vector<ForwardDetLayer const*> const& tecNegative =
      tmpTkGeometry.isThere(GeomDetEnumerators::P2OTEC) ? gsTracker->negTidLayers() : gsTracker->negTecLayers();

  // Get suitable propagators
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(tmpPropagatorAlong, alongMomentum);
  std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(tmpPropagatorOpposite, oppositeToMomentum);

  // Stepping Helix Propagator for propogation from muon system to tracker
  edm::ESHandle<Propagator> shpOpposite = iEventSetup.getHandle(t_SHPOpposite_);

  // Loop over the L2's and make seeds for all of them
  LogTrace(theCategory_) << "TSGForOIDNN::produce: Number of L2's: " << l2TrackCol.size();

  for (auto const& l2 : l2TrackCol) {
    // Container of Seeds
    std::vector<TrajectorySeed> out;
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: L2 muon pT, eta, phi --> " << l2.pt() << " , " << l2.eta()
                            << " , " << l2.phi();

    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(l2, &magfield);

    dummyPlane->move(fts.position() - dummyPlane->position());
    TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatIP: " << tsosAtIP;

    // Get the TSOS on the innermost layer of the L2
    TrajectoryStateOnSurface tsosAtMuonSystem = trajectoryStateTransform::innerStateOnSurface(l2, geometry, &magfield);
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatMuonSystem: " << tsosAtMuonSystem;

    LogTrace("TSGForOIDNN")
        << "TSGForOIDNN::produce: Check the error of the L2 parameter and use hit seeds if big errors";

    StateOnTrackerBound fromInside(propagatorAlong.get());
    TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);

    StateOnTrackerBound fromOutside(&*shpOpposite);
    TrajectoryStateOnSurface outerTkStateOutside = fromOutside(tsosAtMuonSystem);

    // Check if the two positions (using updated and not-updated TSOS) agree withing certain extent.
    // If both TSOSs agree, use only the one at vertex, as it uses more information. If they do not agree, search for seeds based on both.
    double l2muonEta = l2.eta();
    double absL2muonEta = std::abs(l2muonEta);

    // make non-const copies of parameters, so they can be overriden for individual L2 muons
    unsigned int maxHitSeeds = maxHitSeeds_;
    unsigned int maxHitDoubletSeeds = maxHitDoubletSeeds_;
    unsigned int maxHitlessSeedsIP = maxHitlessSeedsIP_;
    unsigned int maxHitlessSeedsMuS = maxHitlessSeedsMuS_;

    float errorSFHitless = fixedErrorRescalingForHitless_;

    // update strategy parameters by evaluating DNN
    if (getStrategyFromDNN_) {
      bool dnnSuccess = false;

      // Update feature map with parameters of the current muon
      updateFeatureMap(feature_map, l2, tsosAtIP, outerTkStateOutside);

      if (useRegressor_) {
        // Use regressor
        evaluateRegressor(feature_map,
                          tf_session_HB_,
                          metadata_.get_child("HB"),
                          tf_session_HLIP_,
                          metadata_.get_child("HLIP"),
                          tf_session_HLMuS_,
                          metadata_.get_child("HLMuS"),
                          strPars,
                          dnnSuccess);
      } else {
        // Use classifier
        evaluateClassifier(feature_map, tf_session_, metadata_, strPars, dnnSuccess);
      }
      if (!dnnSuccess)
        break;

      maxHitSeeds = 0;
      maxHitDoubletSeeds = strPars.nHBd;
      maxHitlessSeedsIP = strPars.nHLIP;
      maxHitlessSeedsMuS = strPars.nHLMuS;
      errorSFHitless = strPars.sf;
    }

    numSeedsMade = 0;
    hitlessSeedsMadeIP = 0;
    hitlessSeedsMadeMuS = 0;
    hitSeedsMade = 0;
    hitDoubletSeedsMade = 0;

    auto createSeeds = [&](auto const& layers) {
      for (auto const& layer : boost::adaptors::reverse(layers)) {
        if (hitlessSeedsMadeIP < maxHitlessSeedsIP && numSeedsMade < maxSeeds_)
          makeSeedsWithoutHits(*layer,
                               tsosAtIP,
                               *(propagatorAlong.get()),
                               estimator,
                               errorSFHitless,
                               hitlessSeedsMadeIP,
                               numSeedsMade,
                               out);

        if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() && hitlessSeedsMadeMuS < maxHitlessSeedsMuS &&
            numSeedsMade < maxSeeds_)
          makeSeedsWithoutHits(*layer,
                               outerTkStateOutside,
                               *(propagatorOpposite.get()),
                               estimator,
                               errorSFHitless,
                               hitlessSeedsMadeMuS,
                               numSeedsMade,
                               out);

        if (hitSeedsMade < maxHitSeeds && numSeedsMade < maxSeeds_)
          makeSeedsFromHits(*layer,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimator,
                            measurementTracker,
                            hitSeedsMade,
                            numSeedsMade,
                            maxHitSeeds,
                            layerCount,
                            out);

        if (hitDoubletSeedsMade < maxHitDoubletSeeds && numSeedsMade < maxSeeds_)
          makeSeedsFromHitDoublets(*layer,
                                   tsosAtIP,
                                   *(propagatorAlong.get()),
                                   estimator,
                                   measurementTracker,
                                   navSchool,
                                   hitDoubletSeedsMade,
                                   numSeedsMade,
                                   maxHitDoubletSeeds,
                                   layerCount,
                                   out);
      };
    };

    // BARREL
    if (absL2muonEta < maxEtaForTOB_) {
      layerCount = 0;
      createSeeds(tob);
      LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                              << " , layerCount = " << layerCount;
    }

    // Reset number of seeds if in overlap region
    if (absL2muonEta > minEtaForTEC_ && absL2muonEta < maxEtaForTOB_) {
      numSeedsMade = 0;
      hitlessSeedsMadeIP = 0;
      hitlessSeedsMadeMuS = 0;
      hitSeedsMade = 0;
      hitDoubletSeedsMade = 0;
    }

    // ENDCAP+
    if (l2muonEta > minEtaForTEC_) {
      layerCount = 0;
      createSeeds(tecPositive);
      LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                              << " , layerCount = " << layerCount;
    }

    // ENDCAP-
    if (l2muonEta < -minEtaForTEC_) {
      layerCount = 0;
      createSeeds(tecNegative);
      LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                              << " , layerCount = " << layerCount;
    }

    for (std::vector<TrajectorySeed>::iterator it = out.begin(); it != out.end(); ++it) {
      result->push_back(*it);
    }

  }  // L2Collection

  edm::LogInfo(theCategory_) << "TSGForOIDNN::produce: number of seeds made: " << result->size();

  iEvent.put(std::move(result));
}

//
// Create seeds without hits on a given layer (TOB or TEC)
//
void TSGForOIDNN::makeSeedsWithoutHits(const GeometricSearchDet& layer,
                                       const TrajectoryStateOnSurface& tsos,
                                       const Propagator& propagatorAlong,
                                       const Chi2MeasurementEstimatorBase& estimator,
                                       double errorSF,
                                       unsigned int& hitlessSeedsMade,
                                       unsigned int& numSeedsMade,
                                       std::vector<TrajectorySeed>& out) const {
  // create hitless seeds
  LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: Start hitless";
  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(tsos, propagatorAlong, estimator, dets);
  if (!dets.empty()) {
    auto const& detOnLayer = dets.front().first;
    auto& tsosOnLayer = dets.front().second;
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: tsosOnLayer " << tsosOnLayer;
    if (!tsosOnLayer.isValid()) {
      edm::LogInfo(theCategory_) << "ERROR!: Hitless TSOS is not valid!";
    } else {
      tsosOnLayer.rescaleError(errorSF);
      PTrajectoryStateOnDet const& ptsod =
          trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
      TrajectorySeed::RecHitContainer rHC;
      out.push_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
      LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: TSOS (Hitless) done ";
      hitlessSeedsMade++;
      numSeedsMade++;
    }
  }
}

//
// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
//
void TSGForOIDNN::makeSeedsFromHits(const GeometricSearchDet& layer,
                                    const TrajectoryStateOnSurface& tsos,
                                    const Propagator& propagatorAlong,
                                    const Chi2MeasurementEstimatorBase& estimator,
                                    const MeasurementTrackerEvent& measurementTracker,
                                    unsigned int& hitSeedsMade,
                                    unsigned int& numSeedsMade,
                                    const unsigned int& maxHitSeeds,
                                    unsigned int& layerCount,
                                    std::vector<TrajectorySeed>& out) const {
  if (layerCount > numOfLayersToTry_)
    return;

  const TrajectoryStateOnSurface& onLayer(tsos);

  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(onLayer, propagatorAlong, estimator, dets);

  // Find Measurements on each DetWithState
  LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Find measurements on each detWithState  " << dets.size();
  std::vector<TrajectoryMeasurement> meas;
  for (auto const& detI : dets) {
    MeasurementDetWithData det = measurementTracker.idToDet(detI.first->geographicalId());
    if (det.isNull())
      continue;
    if (!detI.second.isValid())
      continue;  // Skip if TSOS is not valid

    std::vector<TrajectoryMeasurement> mymeas =
        det.fastMeasurements(detI.second, onLayer, propagatorAlong, estimator);  // Second TSOS is not used
    for (auto const& measurement : mymeas) {
      if (measurement.recHit()->isValid())
        meas.push_back(measurement);  // Only save those which are valid
    }
  }

  // Update TSOS using TMs after sorting, then create Trajectory Seed and put into vector
  LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Update TSOS using TMs after sorting, then create "
                             "Trajectory Seed, number of TM = "
                          << meas.size();
  std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

  unsigned int found = 0;
  for (auto const& measurement : meas) {
    if (hitSeedsMade >= maxHitSeeds)
      return;
    TrajectoryStateOnSurface updatedTSOS = updator_->update(measurement.forwardPredictedState(), *measurement.recHit());
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: TSOS for TM " << found;
    if (not updatedTSOS.isValid())
      continue;

    edm::OwnVector<TrackingRecHit> seedHits;
    seedHits.push_back(*measurement.recHit()->hit());
    PTrajectoryStateOnDet const& pstate =
        trajectoryStateTransform::persistentState(updatedTSOS, measurement.recHit()->geographicalId().rawId());
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Number of seedHits: " << seedHits.size();
    TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);
    out.push_back(seed);
    found++;
    numSeedsMade++;
    hitSeedsMade++;
    if (found == numOfHitsToTry_)
      break;
  }

  if (found)
    layerCount++;
}

//
// Find hits compatible with L2 trajectory on two adjacent layers; if found, create a seed using both hits
//
void TSGForOIDNN::makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                           const TrajectoryStateOnSurface& tsos,
                                           const Propagator& propagatorAlong,
                                           const Chi2MeasurementEstimatorBase& estimator,
                                           const MeasurementTrackerEvent& measurementTracker,
                                           const NavigationSchool& navSchool,
                                           unsigned int& hitDoubletSeedsMade,
                                           unsigned int& numSeedsMade,
                                           const unsigned int& maxHitDoubletSeeds,
                                           unsigned int& layerCount,
                                           std::vector<TrajectorySeed>& out) const {
  // This method is similar to makeSeedsFromHits, but the seed is created
  // only when in addition to a hit on a given layer, there are more compatible hits
  // on next layers (going from outside inwards), compatible with updated TSOS.
  // If that's the case, multiple compatible hits are used to create a single seed.

  // Configured to only check the immideately adjacent layer and add one more hit
  int max_addtnl_layers = 1;  // max number of additional layers to scan
  int max_meas = 1;           // number of measurements to consider on each additional layer

  // // // First, regular procedure to find a compatible hit - like in makeSeedsFromHits // // //

  const TrajectoryStateOnSurface& onLayer(tsos);

  // Find dets compatible with original TSOS
  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(onLayer, propagatorAlong, estimator, dets);

  LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Find measurements on each detWithState  "
                          << dets.size();
  std::vector<TrajectoryMeasurement> meas;

  // Loop over dets
  for (auto const& detI : dets) {
    MeasurementDetWithData det = measurementTracker.idToDet(detI.first->geographicalId());

    if (det.isNull())
      continue;  // skip if det does not exist
    if (!detI.second.isValid())
      continue;  // skip if TSOS is invalid

    // Find measurements on this det
    std::vector<TrajectoryMeasurement> mymeas = det.fastMeasurements(detI.second, onLayer, propagatorAlong, estimator);

    // Save valid measurements
    for (auto const& measurement : mymeas) {
      if (measurement.recHit()->isValid())
        meas.push_back(measurement);
    }  // end loop over meas
  }    // end loop over dets

  LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Update TSOS using TMs after sorting, then create "
                             "Trajectory Seed, number of TM = "
                          << meas.size();

  // sort valid measurements found on the first layer
  std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

  unsigned int found = 0;
  int hit_num = 0;

  // Loop over all valid measurements compatible with original TSOS
  //for (std::vector<TrajectoryMeasurement>::const_iterator mea = meas.begin(); mea != meas.end(); ++mea) {
  for (auto const& measurement : meas) {
    if (hitDoubletSeedsMade >= maxHitDoubletSeeds)
      return;  // abort if enough seeds created

    hit_num++;

    // Update TSOS with measurement on first considered layer
    TrajectoryStateOnSurface updatedTSOS = updator_->update(measurement.forwardPredictedState(), *measurement.recHit());

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: TSOS for TM " << found;
    if (not updatedTSOS.isValid())
      continue;  // Skip if updated TSOS is invalid

    edm::OwnVector<TrackingRecHit> seedHits;

    // Save hit on first layer
    seedHits.push_back(*measurement.recHit()->hit());
    const DetLayer* detLayer = dynamic_cast<const DetLayer*>(&layer);

    // // // Now for this measurement we will loop over additional layers and try to update the TSOS again // // //

    // find layers compatible with updated TSOS
    auto const& compLayers = navSchool.nextLayers(*detLayer, *updatedTSOS.freeState(), alongMomentum);

    int addtnl_layers_scanned = 0;
    int found_compatible_on_next_layer = 0;
    int det_id = 0;

    // Copy updated TSOS - we will update it again with a measurement from the next layer, if we find it
    TrajectoryStateOnSurface updatedTSOS_next(updatedTSOS);

    // loop over layers compatible with updated TSOS
    for (auto compLayer : compLayers) {
      int nmeas = 0;

      if (addtnl_layers_scanned >= max_addtnl_layers)
        break;  // break if we already looped over enough layers
      if (found_compatible_on_next_layer > 0)
        break;  // break if we already found additional hit

      // find dets compatible with updated TSOS
      std::vector<GeometricSearchDet::DetWithState> dets_next;
      TrajectoryStateOnSurface onLayer_next(updatedTSOS);

      compLayer->compatibleDetsV(onLayer_next, propagatorAlong, estimator, dets_next);

      //if (!detWithState.size()) continue;
      std::vector<TrajectoryMeasurement> meas_next;

      // find measurements on dets_next and save the valid ones
      for (auto const& detI_next : dets_next) {
        MeasurementDetWithData det = measurementTracker.idToDet(detI_next.first->geographicalId());

        if (det.isNull())
          continue;  // skip if det does not exist
        if (!detI_next.second.isValid())
          continue;  // skip if TSOS is invalid

        // Find measurements on this det
        std::vector<TrajectoryMeasurement> mymeas_next =
            det.fastMeasurements(detI_next.second, onLayer_next, propagatorAlong, estimator);

        for (auto const& mea_next : mymeas_next) {
          // save valid measurements
          if (mea_next.recHit()->isValid())
            meas_next.push_back(mea_next);

        }  // end loop over mymeas_next
      }    // end loop over dets_next

      // sort valid measurements found on this layer
      std::sort(meas_next.begin(), meas_next.end(), TrajMeasLessEstim());

      // loop over valid measurements compatible with updated TSOS (TSOS updated with a hit on the first layer)
      for (auto const& mea_next : meas_next) {
        if (nmeas >= max_meas)
          break;  // skip if we already found enough hits

        // try to update TSOS again, with an additional hit
        updatedTSOS_next = updator_->update(mea_next.forwardPredictedState(), *mea_next.recHit());

        if (not updatedTSOS_next.isValid())
          continue;  // skip if TSOS updated with additional hit is not valid

        // If there was a compatible hit on this layer, we end up here.
        // An additional compatible hit is saved.
        seedHits.push_back(*mea_next.recHit()->hit());
        det_id = mea_next.recHit()->geographicalId().rawId();
        nmeas++;
        found_compatible_on_next_layer++;

      }  // end loop over meas_next

      addtnl_layers_scanned++;

    }  // end loop over compLayers (additional layers scanned after the original layer)

    if (found_compatible_on_next_layer == 0)
      continue;
    // only consider the hit if there was a compatible hit on one of the additional scanned layers

    // Create a seed from two saved hits
    PTrajectoryStateOnDet const& pstate = trajectoryStateTransform::persistentState(updatedTSOS_next, det_id);
    TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Number of seedHits: " << seedHits.size();
    out.push_back(seed);

    found++;
    numSeedsMade++;
    hitDoubletSeedsMade++;

    if (found == numOfHitsToTry_)
      break;  // break if enough measurements scanned

  }  // end loop over measurements compatible with original TSOS

  if (found)
    layerCount++;
}

//
// Update the dictionary of variables to use as input features for DNN
//
void TSGForOIDNN::updateFeatureMap(std::unordered_map<std::string, float>& the_map,
                                   const reco::Track& l2,
                                   const TrajectoryStateOnSurface& tsos_IP,
                                   const TrajectoryStateOnSurface& tsos_MuS) const {
  the_map["pt"] = l2.pt();
  the_map["eta"] = l2.eta();
  the_map["phi"] = l2.phi();
  the_map["validHits"] = l2.found();
  if (tsos_IP.isValid()) {
    the_map["tsos_IP_eta"] = tsos_IP.globalPosition().eta();
    the_map["tsos_IP_phi"] = tsos_IP.globalPosition().phi();
    the_map["tsos_IP_pt"] = tsos_IP.globalMomentum().perp();
    the_map["tsos_IP_pt_eta"] = tsos_IP.globalMomentum().eta();
    the_map["tsos_IP_pt_phi"] = tsos_IP.globalMomentum().phi();
    const AlgebraicSymMatrix55& matrix_IP = tsos_IP.curvilinearError().matrix();
    the_map["err0_IP"] = sqrt(matrix_IP[0][0]);
    the_map["err1_IP"] = sqrt(matrix_IP[1][1]);
    the_map["err2_IP"] = sqrt(matrix_IP[2][2]);
    the_map["err3_IP"] = sqrt(matrix_IP[3][3]);
    the_map["err4_IP"] = sqrt(matrix_IP[4][4]);
    the_map["tsos_IP_valid"] = 1.0;
  } else {
    the_map["tsos_IP_eta"] = -999;
    the_map["tsos_IP_phi"] = -999;
    the_map["tsos_IP_pt"] = -999;
    the_map["tsos_IP_pt_eta"] = -999;
    the_map["tsos_IP_pt_phi"] = -999;
    the_map["err0_IP"] = -999;
    the_map["err1_IP"] = -999;
    the_map["err2_IP"] = -999;
    the_map["err3_IP"] = -999;
    the_map["err4_IP"] = -999;
    the_map["tsos_IP_valid"] = 0.0;
  }
  if (tsos_MuS.isValid()) {
    the_map["tsos_MuS_eta"] = tsos_MuS.globalPosition().eta();
    the_map["tsos_MuS_phi"] = tsos_MuS.globalPosition().phi();
    the_map["tsos_MuS_pt"] = tsos_MuS.globalMomentum().perp();
    the_map["tsos_MuS_pt_eta"] = tsos_MuS.globalMomentum().eta();
    the_map["tsos_MuS_pt_phi"] = tsos_MuS.globalMomentum().phi();
    const AlgebraicSymMatrix55& matrix_MuS = tsos_MuS.curvilinearError().matrix();
    the_map["err0_MuS"] = sqrt(matrix_MuS[0][0]);
    the_map["err1_MuS"] = sqrt(matrix_MuS[1][1]);
    the_map["err2_MuS"] = sqrt(matrix_MuS[2][2]);
    the_map["err3_MuS"] = sqrt(matrix_MuS[3][3]);
    the_map["err4_MuS"] = sqrt(matrix_MuS[4][4]);
    the_map["tsos_MuS_valid"] = 1.0;
  } else {
    the_map["tsos_MuS_eta"] = -999;
    the_map["tsos_MuS_phi"] = -999;
    the_map["tsos_MuS_pt"] = -999;
    the_map["tsos_MuS_pt_eta"] = -999;
    the_map["tsos_MuS_pt_phi"] = -999;
    the_map["err0_MuS"] = -999;
    the_map["err1_MuS"] = -999;
    the_map["err2_MuS"] = -999;
    the_map["err3_MuS"] = -999;
    the_map["err4_MuS"] = -999;
    the_map["tsos_MuS_valid"] = 0.0;
  }
}

//
// Obtain seeding strategy parameters by evaluating DNN classifier for a given muon
//
void TSGForOIDNN::evaluateClassifier(const std::unordered_map<std::string, float>& feature_map,
                                     tensorflow::Session* session,
                                     const pt::ptree& metadata,
                                     StrategyParameters& out,
                                     bool& dnnSuccess) const {
  int n_features = metadata.get<int>("n_features", 0);

  // Prepare tensor for DNN inputs
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, n_features});
  std::string fname;
  int i_feature = 0;
  for (const pt::ptree::value_type& feature : metadata.get_child("feature_names")) {
    fname = feature.second.data();
    if (feature_map.find(fname) == feature_map.end()) {
      // don't evaluate DNN if any input feature is missing
      dnnSuccess = false;
    } else {
      input.matrix<float>()(0, i_feature) = float(feature_map.at(fname));
      i_feature++;
    }
  }

  // Prepare tensor for DNN outputs
  std::vector<tensorflow::Tensor> outputs;

  // Evaluate DNN and put results in output tensor
  std::string inputLayer = metadata.get<std::string>("input_layer");
  std::string outputLayer = metadata.get<std::string>("output_layer");

  tensorflow::run(session, {{inputLayer, input}}, {outputLayer}, &outputs);
  tensorflow::Tensor out_tensor = outputs[0];
  tensorflow::TTypes<float, 1>::Matrix dnn_outputs = out_tensor.matrix<float>();

  // Find output with largest prediction
  int imax = 0;
  float out_max = 0;
  for (long long int i = 0; i < out_tensor.dim_size(1); i++) {
    float ith_output = dnn_outputs(0, i);
    if (ith_output > out_max) {
      imax = i;
      out_max = ith_output;
    }
  }

  // Decode output
  const std::string label = "output_labels.label_" + std::to_string(imax);
  out.nHBd = metadata.get<int>(label + ".nHBd");
  out.nHLIP = metadata.get<int>(label + ".nHLIP");
  out.nHLMuS = metadata.get<int>(label + ".nHLMuS");
  out.sf = metadata.get<int>(label + ".SF");

  dnnSuccess = true;
}

//
// Obtain seeding strategy parameters by evaluating DNN regressor for a given muon
//
void TSGForOIDNN::evaluateRegressor(const std::unordered_map<std::string, float>& feature_map,
                                    tensorflow::Session* session_HB,
                                    const pt::ptree& metadata_HB,
                                    tensorflow::Session* session_HLIP,
                                    const pt::ptree& metadata_HLIP,
                                    tensorflow::Session* session_HLMuS,
                                    const pt::ptree& metadata_HLMuS,
                                    StrategyParameters& out,
                                    bool& dnnSuccess) const {
  int n_features = metadata_HB.get<int>("n_features", 0);

  // Prepare tensor for DNN inputs
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, n_features});
  std::string fname;
  int i_feature = 0;
  for (const pt::ptree::value_type& feature : metadata_HB.get_child("feature_names")) {
    fname = feature.second.data();
    if (feature_map.find(fname) == feature_map.end()) {
      // don't evaluate DNN if any input feature is missing
      dnnSuccess = false;
    } else {
      input.matrix<float>()(0, i_feature) = float(feature_map.at(fname));
      i_feature++;
    }
  }

  // Prepare tensor for DNN outputs
  std::vector<tensorflow::Tensor> outputs_HB;
  // Evaluate DNN and put results in output tensor
  std::string inputLayer_HB = metadata_HB.get<std::string>("input_layer");
  std::string outputLayer_HB = metadata_HB.get<std::string>("output_layer");
  tensorflow::run(session_HB, {{inputLayer_HB, input}}, {outputLayer_HB}, &outputs_HB);
  tensorflow::Tensor out_tensor_HB = outputs_HB[0];
  tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HB = out_tensor_HB.matrix<float>();

  // Prepare tensor for DNN outputs
  std::vector<tensorflow::Tensor> outputs_HLIP;
  // Evaluate DNN and put results in output tensor
  std::string inputLayer_HLIP = metadata_HLIP.get<std::string>("input_layer");
  std::string outputLayer_HLIP = metadata_HLIP.get<std::string>("output_layer");
  tensorflow::run(session_HLIP, {{inputLayer_HLIP, input}}, {outputLayer_HLIP}, &outputs_HLIP);
  tensorflow::Tensor out_tensor_HLIP = outputs_HLIP[0];
  tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HLIP = out_tensor_HLIP.matrix<float>();

  // Prepare tensor for DNN outputs
  std::vector<tensorflow::Tensor> outputs_HLMuS;
  // Evaluate DNN and put results in output tensor
  std::string inputLayer_HLMuS = metadata_HLMuS.get<std::string>("input_layer");
  std::string outputLayer_HLMuS = metadata_HLMuS.get<std::string>("output_layer");
  tensorflow::run(session_HLMuS, {{inputLayer_HLMuS, input}}, {outputLayer_HLMuS}, &outputs_HLMuS);
  tensorflow::Tensor out_tensor_HLMuS = outputs_HLMuS[0];
  tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HLMuS = out_tensor_HLMuS.matrix<float>();

  // Decode output
  out.nHBd = round(dnn_outputs_HB(0, 0));
  out.nHLIP = round(dnn_outputs_HLIP(0, 0));
  out.sf = round(dnn_outputs_HLIP(0, 1));
  out.nHLMuS = round(dnn_outputs_HLMuS(0, 0));

  // Prevent prediction of negative number of seeds or too many seeds
  out.nHBd = std::clamp(out.nHBd, 0, 10);
  out.nHLIP = std::clamp(out.nHLIP, 0, 10);
  out.nHLMuS = std::clamp(out.nHLMuS, 0, 10);

  // Prevent prediction of 0 seeds in total
  if (out.nHBd == 0 && out.nHLIP == 0 && out.nHLMuS == 0) {
    // default strategy, similar to Run 2
    out.nHBd = 1;
    out.nHLIP = 5;
  }

  // Prevent extreme predictions for scale factors
  // (on average SF=2 was found to be optimal)
  if (out.sf <= 0)
    out.sf = 2;
  if (out.sf > 10)
    out.sf = 10;

  dnnSuccess = true;
}

//
// Default values of configuration parameters
//
void TSGForOIDNN::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  desc.add<int>("layersToTry", 2);
  desc.add<double>("fixedErrorRescaleFactorForHitless", 2.0);
  desc.add<int>("hitsToTry", 1);
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag("hltSiStripClusters"));
  desc.add<std::string>("estimator", "hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB", 1.8);
  desc.add<double>("minEtaForTEC", 0.7);
  desc.addUntracked<bool>("debug", false);
  desc.add<unsigned int>("maxSeeds", 20);
  desc.add<unsigned int>("maxHitlessSeeds", 5);
  desc.add<unsigned int>("maxHitSeeds", 1);
  desc.add<std::string>("propagatorName", "PropagatorWithMaterialParabolicMf");
  desc.add<unsigned int>("maxHitlessSeedsIP", 5);
  desc.add<unsigned int>("maxHitlessSeedsMuS", 0);
  desc.add<unsigned int>("maxHitDoubletSeeds", 0);
  desc.add<bool>("getStrategyFromDNN", false);
  desc.add<bool>("useRegressor", false);
  desc.add<std::string>("dnnMetadataPath", "");
  descriptions.add("tsgForOIDNN", desc);
}

DEFINE_FWK_MODULE(TSGForOIDNN);
