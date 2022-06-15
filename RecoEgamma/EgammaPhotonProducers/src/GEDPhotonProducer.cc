/** \class GEDPhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMVABasedHaloTagger.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EgammaLocalCovParamDefaults.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonDNNEstimator.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EcalPFClusterIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

class CacheData {
public:
  CacheData(const edm::ParameterSet& conf) {
    // Here we will have to load the DNN PFID if present in the config
    egammaTools::DNNConfiguration config;
    const auto& pset_dnn = conf.getParameter<edm::ParameterSet>("PhotonDNNPFid");
    const auto dnnEnabled = pset_dnn.getParameter<bool>("enabled");
    if (dnnEnabled) {
      config.inputTensorName = pset_dnn.getParameter<std::string>("inputTensorName");
      config.outputTensorName = pset_dnn.getParameter<std::string>("outputTensorName");
      config.modelsFiles = pset_dnn.getParameter<std::vector<std::string>>("modelsFiles");
      config.scalersFiles = pset_dnn.getParameter<std::vector<std::string>>("scalersFiles");
      config.outputDim = pset_dnn.getParameter<std::vector<unsigned int>>("outputDim");
      const auto useEBModelInGap = pset_dnn.getParameter<bool>("useEBModelInGap");
      photonDNNEstimator = std::make_unique<PhotonDNNEstimator>(config, useEBModelInGap);
    }
    ///for MVA based beam halo tagger in the EE
    const auto runMVABasedHaloTagger = conf.getParameter<bool>("runMVABasedHaloTagger");
    edm::ParameterSet mvaBasedHaloVariableSet = conf.getParameter<edm::ParameterSet>("mvaBasedHaloVariableSet");
    auto trainingFileName_ = mvaBasedHaloVariableSet.getParameter<edm::FileInPath>(("trainingFileName")).fullPath();
    if (runMVABasedHaloTagger) {
      haloTaggerGBR = createGBRForest(trainingFileName_);
    }
  }
  std::unique_ptr<const PhotonDNNEstimator> photonDNNEstimator;
  std::unique_ptr<const GBRForest> haloTaggerGBR;
};

class GEDPhotonProducer : public edm::stream::EDProducer<edm::GlobalCache<CacheData>> {
public:
  GEDPhotonProducer(const edm::ParameterSet& ps, const CacheData* gcache);

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const CacheData*){};

  void endStream() override;

private:
  class RecoStepInfo {
  public:
    enum FlagBits { kOOT = 0x1, kFinal = 0x2 };
    explicit RecoStepInfo(const std::string& recoStep);

    bool isOOT() const { return flags_ & kOOT; }
    bool isFinal() const { return flags_ & kFinal; }

  private:
    unsigned int flags_;
  };

  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                            const CaloTopology* topology,
                            const EcalRecHitCollection* ecalBarrelHits,
                            const EcalRecHitCollection* ecalEndcapHits,
                            const EcalRecHitCollection* preshowerHits,
                            const ElectronHcalHelper* hcalHelperCone,
                            const ElectronHcalHelper* hcalHelperBc,
                            const reco::VertexCollection& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC,
                            EcalPFRecHitThresholds const& thresholds);

  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCollection>& photonHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
                            reco::VertexCollection const& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadrons,
                            const edm::Handle<edm::ValueMap<float>>& neutralHadrons,
                            const edm::Handle<edm::ValueMap<float>>& photons,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtx,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtxGeomVeto,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsPFPV,
                            const edm::Handle<edm::ValueMap<float>>& pfEcalClusters,
                            const edm::Handle<edm::ValueMap<float>>& pfHcalClusters);

  // std::string PhotonCoreCollection_;
  std::string photonCollection_;
  const edm::InputTag photonProducer_;

  edm::EDGetTokenT<reco::PhotonCoreCollection> photonCoreProducerT_;
  edm::EDGetTokenT<reco::PhotonCollection> photonProducerT_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> preshowerHits_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits_;
  edm::EDGetTokenT<reco::VertexCollection> vertexProducer_;
  //for isolation with map-based veto
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationToken;
  //photon isolation sums
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoNeutralHadronIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoPhotonIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedWorstVtxIsoToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedWorstVtxGeomVetoIsoToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedPFPVIsoToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> phoPFECALClusIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoPFHCALClusIsolationToken_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterESGetTokens_;

  std::string valueMapPFCandPhoton_;

  std::unique_ptr<PhotonIsolationCalculator> photonIsoCalculator_ = nullptr;

  //AA
  //Flags and severities to be excluded from calculations

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double multThresEB_;
  double multThresEE_;
  double hOverEConeSize_;
  bool checkHcalStatus_;
  double highEt_;
  double minR9Barrel_;
  double minR9Endcap_;
  bool runMIPTagger_;
  bool runMVABasedHaloTagger_;

  RecoStepInfo recoStep_;

  bool usePrimaryVertex_;

  CaloGeometry const* caloGeom_ = nullptr;

  //MIP
  std::unique_ptr<PhotonMIPHaloTagger> photonMIPHaloTagger_ = nullptr;
  //MVA based Halo tagger for the EE photons
  std::unique_ptr<PhotonMVABasedHaloTagger> photonMVABasedHaloTagger_ = nullptr;

  std::vector<double> preselCutValuesBarrel_;
  std::vector<double> preselCutValuesEndcap_;

  std::unique_ptr<PhotonEnergyCorrector> photonEnergyCorrector_ = nullptr;
  std::string candidateP4type_;

  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  const edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRechitThresholdsToken_;

  // additional configuration and helpers
  std::unique_ptr<ElectronHcalHelper> hcalHelperCone_;
  std::unique_ptr<ElectronHcalHelper> hcalHelperBc_;
  bool hcalRun2EffDepth_;

  // DNN for PFID photon enabled
  bool dnnPFidEnabled_;
  std::vector<tensorflow::Session*> tfSessions_;

  double ecaldrMax_;
  double ecaldrVetoBarrel_;
  double ecaldrVetoEndcap_;
  double ecaletaStripBarrel_;
  double ecaletaStripEndcap_;
  double ecalenergyBarrel_;
  double ecalenergyEndcap_;
  typedef EcalPFClusterIsolation<reco::Photon> PhotonEcalPFClusterIsolation;
  std::unique_ptr<PhotonEcalPFClusterIsolation> ecalisoAlgo = nullptr;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducer_;

  bool useHF_;
  double hcaldrMax_;
  double hcaldrVetoBarrel_;
  double hcaldrVetoEndcap_;
  double hcaletaStripBarrel_;
  double hcaletaStripEndcap_;
  double hcalenergyBarrel_;
  double hcalenergyEndcap_;
  double hcaluseEt_;

  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHCAL_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFEM_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFHAD_;

  typedef HcalPFClusterIsolation<reco::Photon> PhotonHcalPFClusterIsolation;
  std::unique_ptr<PhotonHcalPFClusterIsolation> hcalisoAlgo = nullptr;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDPhotonProducer);

namespace {
  inline double ptFast(const double energy, const math::XYZPoint& position, const math::XYZPoint& origin) {
    const auto v = position - origin;
    return energy * std::sqrt(v.perp2() / v.mag2());
  }
}  // namespace

GEDPhotonProducer::RecoStepInfo::RecoStepInfo(const std::string& step) : flags_(0) {
  if (step == "final")
    flags_ = kFinal;
  else if (step == "oot")
    flags_ = kOOT;
  else if (step == "ootfinal")
    flags_ = (kOOT | kFinal);
  else if (step == "tmp")
    flags_ = 0;
  else {
    throw cms::Exception("InvalidConfig")
        << " reconstructStep " << step << " is invalid, the options are: tmp, final,oot or ootfinal" << std::endl;
  }
}

GEDPhotonProducer::GEDPhotonProducer(const edm::ParameterSet& config, const CacheData* gcache)
    : photonProducer_{config.getParameter<edm::InputTag>("photonProducer")},
      ecalClusterESGetTokens_{consumesCollector()},
      recoStep_(config.getParameter<std::string>("reconstructionStep")),
      caloTopologyToken_{esConsumes()},
      caloGeometryToken_{esConsumes()},
      ecalPFRechitThresholdsToken_{esConsumes()},
      hcalHelperCone_(nullptr),
      hcalHelperBc_(nullptr) {
  if (recoStep_.isFinal()) {
    photonProducerT_ = consumes(photonProducer_);
    pfCandidates_ = consumes(config.getParameter<edm::InputTag>("pfCandidates"));

    const edm::ParameterSet& pfIsolCfg = config.getParameter<edm::ParameterSet>("pfIsolCfg");
    auto getVMToken = [&pfIsolCfg, this](const std::string& name) {
      return consumes(pfIsolCfg.getParameter<edm::InputTag>(name));
    };
    phoChargedIsolationToken_ = getVMToken("chargedHadronIso");
    phoNeutralHadronIsolationToken_ = getVMToken("neutralHadronIso");
    phoPhotonIsolationToken_ = getVMToken("photonIso");
    phoChargedWorstVtxIsoToken_ = getVMToken("chargedHadronWorstVtxIso");
    phoChargedWorstVtxGeomVetoIsoToken_ = getVMToken("chargedHadronWorstVtxGeomVetoIso");
    phoChargedPFPVIsoToken_ = getVMToken("chargedHadronPFPVIso");

    //OOT photons in legacy 80X re-miniAOD do not have PF cluster embeded into the reco object
    //to preserve 80X behaviour
    if (config.exists("pfECALClusIsolation")) {
      phoPFECALClusIsolationToken_ = consumes(config.getParameter<edm::InputTag>("pfECALClusIsolation"));
    }
    if (config.exists("pfHCALClusIsolation")) {
      phoPFHCALClusIsolationToken_ = consumes(config.getParameter<edm::InputTag>("pfHCALClusIsolation"));
    }

  } else {
    photonCoreProducerT_ = consumes(photonProducer_);
  }

  auto pfEg = config.getParameter<edm::InputTag>("pfEgammaCandidates");
  if (not pfEg.label().empty()) {
    pfEgammaCandidates_ = consumes(pfEg);
  }
  barrelEcalHits_ = consumes(config.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_ = consumes(config.getParameter<edm::InputTag>("endcapEcalHits"));
  preshowerHits_ = consumes(config.getParameter<edm::InputTag>("preshowerHits"));
  vertexProducer_ = consumes(config.getParameter<edm::InputTag>("primaryVertexProducer"));

  auto hbhetag = config.getParameter<edm::InputTag>("hbheRecHits");
  if (not hbhetag.label().empty())
    hbheRecHits_ = consumes<HBHERecHitCollection>(hbhetag);

  //
  photonCollection_ = config.getParameter<std::string>("outputPhotonCollection");
  multThresEB_ = config.getParameter<double>("multThresEB");
  multThresEE_ = config.getParameter<double>("multThresEE");
  hOverEConeSize_ = config.getParameter<double>("hOverEConeSize");
  highEt_ = config.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_ = config.getParameter<double>("minR9Barrel");
  minR9Endcap_ = config.getParameter<double>("minR9Endcap");
  usePrimaryVertex_ = config.getParameter<bool>("usePrimaryVertex");
  runMIPTagger_ = config.getParameter<bool>("runMIPTagger");
  runMVABasedHaloTagger_ = config.getParameter<bool>("runMVABasedHaloTagger");

  candidateP4type_ = config.getParameter<std::string>("candidateP4type");
  valueMapPFCandPhoton_ = config.getParameter<std::string>("valueMapPhotons");

  //AA
  //Flags and Severities to be excluded from photon calculations
  auto const& flagnamesEB = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");
  auto const& flagnamesEE = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);
  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  auto const& severitynamesEB = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEB");
  auto const& severitynamesEE = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEE");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);
  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  photonEnergyCorrector_ = std::make_unique<PhotonEnergyCorrector>(config, consumesCollector());

  checkHcalStatus_ = config.getParameter<bool>("checkHcalStatus");
  if (not hbheRecHits_.isUninitialized()) {
    ElectronHcalHelper::Configuration cfgCone, cfgBc;
    cfgCone.hOverEConeSize = hOverEConeSize_;
    if (cfgCone.hOverEConeSize > 0) {
      cfgCone.onlyBehindCluster = false;
      cfgCone.checkHcalStatus = checkHcalStatus_;

      cfgCone.hbheRecHits = hbheRecHits_;

      cfgCone.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
      cfgCone.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
      cfgCone.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
      cfgCone.maxSeverityHE = cfgCone.maxSeverityHB;
    }
    cfgBc.hOverEConeSize = 0.;
    cfgBc.onlyBehindCluster = true;
    cfgBc.checkHcalStatus = checkHcalStatus_;

    cfgBc.hbheRecHits = hbheRecHits_;

    cfgBc.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
    cfgBc.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
    cfgBc.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
    cfgBc.maxSeverityHE = cfgBc.maxSeverityHB;

    hcalHelperCone_ = std::make_unique<ElectronHcalHelper>(cfgCone, consumesCollector());
    hcalHelperBc_ = std::make_unique<ElectronHcalHelper>(cfgBc, consumesCollector());
  }

  hcalRun2EffDepth_ = config.getParameter<bool>("hcalRun2EffDepth");

  // cut values for pre-selection
  preselCutValuesBarrel_ = {config.getParameter<double>("minSCEtBarrel"),
                            config.getParameter<double>("maxHoverEBarrel"),
                            config.getParameter<double>("ecalRecHitSumEtOffsetBarrel"),
                            config.getParameter<double>("ecalRecHitSumEtSlopeBarrel"),
                            config.getParameter<double>("hcalRecHitSumEtOffsetBarrel"),
                            config.getParameter<double>("hcalRecHitSumEtSlopeBarrel"),
                            config.getParameter<double>("nTrackSolidConeBarrel"),
                            config.getParameter<double>("nTrackHollowConeBarrel"),
                            config.getParameter<double>("trackPtSumSolidConeBarrel"),
                            config.getParameter<double>("trackPtSumHollowConeBarrel"),
                            config.getParameter<double>("sigmaIetaIetaCutBarrel")};
  //
  preselCutValuesEndcap_ = {config.getParameter<double>("minSCEtEndcap"),
                            config.getParameter<double>("maxHoverEEndcap"),
                            config.getParameter<double>("ecalRecHitSumEtOffsetEndcap"),
                            config.getParameter<double>("ecalRecHitSumEtSlopeEndcap"),
                            config.getParameter<double>("hcalRecHitSumEtOffsetEndcap"),
                            config.getParameter<double>("hcalRecHitSumEtSlopeEndcap"),
                            config.getParameter<double>("nTrackSolidConeEndcap"),
                            config.getParameter<double>("nTrackHollowConeEndcap"),
                            config.getParameter<double>("trackPtSumSolidConeEndcap"),
                            config.getParameter<double>("trackPtSumHollowConeEndcap"),
                            config.getParameter<double>("sigmaIetaIetaCutEndcap")};
  //

  //moved from beginRun to here, I dont see how this could cause harm as its just reading in the exactly same parameters each run
  if (!recoStep_.isFinal()) {
    photonIsoCalculator_ = std::make_unique<PhotonIsolationCalculator>();
    edm::ParameterSet isolationSumsCalculatorSet = config.getParameter<edm::ParameterSet>("isolationSumsCalculatorSet");
    photonIsoCalculator_->setup(isolationSumsCalculatorSet,
                                flagsexclEB_,
                                flagsexclEE_,
                                severitiesexclEB_,
                                severitiesexclEE_,
                                consumesCollector());
    photonMIPHaloTagger_ = std::make_unique<PhotonMIPHaloTagger>();
    edm::ParameterSet mipVariableSet = config.getParameter<edm::ParameterSet>("mipVariableSet");
    photonMIPHaloTagger_->setup(mipVariableSet, consumesCollector());
  }

  if (recoStep_.isFinal() && runMVABasedHaloTagger_) {
    edm::ParameterSet mvaBasedHaloVariableSet = config.getParameter<edm::ParameterSet>("mvaBasedHaloVariableSet");
    photonMVABasedHaloTagger_ =
        std::make_unique<PhotonMVABasedHaloTagger>(mvaBasedHaloVariableSet, consumesCollector());
  }

  ///Get the set for PF cluster isolation calculator
  const edm::ParameterSet& pfECALClusIsolCfg = config.getParameter<edm::ParameterSet>("pfECALClusIsolCfg");
  pfClusterProducer_ =
      consumes<reco::PFClusterCollection>(pfECALClusIsolCfg.getParameter<edm::InputTag>("pfClusterProducer"));
  ecaldrMax_ = pfECALClusIsolCfg.getParameter<double>("drMax");
  ecaldrVetoBarrel_ = pfECALClusIsolCfg.getParameter<double>("drVetoBarrel");
  ecaldrVetoEndcap_ = pfECALClusIsolCfg.getParameter<double>("drVetoEndcap");
  ecaletaStripBarrel_ = pfECALClusIsolCfg.getParameter<double>("etaStripBarrel");
  ecaletaStripEndcap_ = pfECALClusIsolCfg.getParameter<double>("etaStripEndcap");
  ecalenergyBarrel_ = pfECALClusIsolCfg.getParameter<double>("energyBarrel");
  ecalenergyEndcap_ = pfECALClusIsolCfg.getParameter<double>("energyEndcap");

  const edm::ParameterSet& pfHCALClusIsolCfg = config.getParameter<edm::ParameterSet>("pfHCALClusIsolCfg");
  pfClusterProducerHCAL_ = consumes(pfHCALClusIsolCfg.getParameter<edm::InputTag>("pfClusterProducerHCAL"));
  pfClusterProducerHFEM_ = consumes(pfHCALClusIsolCfg.getParameter<edm::InputTag>("pfClusterProducerHFEM"));
  pfClusterProducerHFHAD_ = consumes(pfHCALClusIsolCfg.getParameter<edm::InputTag>("pfClusterProducerHFHAD"));
  useHF_ = pfHCALClusIsolCfg.getParameter<bool>("useHF");
  hcaldrMax_ = pfHCALClusIsolCfg.getParameter<double>("drMax");
  hcaldrVetoBarrel_ = pfHCALClusIsolCfg.getParameter<double>("drVetoBarrel");
  hcaldrVetoEndcap_ = pfHCALClusIsolCfg.getParameter<double>("drVetoEndcap");
  hcaletaStripBarrel_ = pfHCALClusIsolCfg.getParameter<double>("etaStripBarrel");
  hcaletaStripEndcap_ = pfHCALClusIsolCfg.getParameter<double>("etaStripEndcap");
  hcalenergyBarrel_ = pfHCALClusIsolCfg.getParameter<double>("energyBarrel");
  hcalenergyEndcap_ = pfHCALClusIsolCfg.getParameter<double>("energyEndcap");
  hcaluseEt_ = pfHCALClusIsolCfg.getParameter<bool>("useEt");

  // Register the product
  produces<reco::PhotonCollection>(photonCollection_);
  if (not pfEgammaCandidates_.isUninitialized()) {
    produces<edm::ValueMap<reco::PhotonRef>>(valueMapPFCandPhoton_);
  }

  const auto& pset_dnn = config.getParameter<edm::ParameterSet>("PhotonDNNPFid");
  dnnPFidEnabled_ = pset_dnn.getParameter<bool>("enabled");
  if (dnnPFidEnabled_) {
    tfSessions_ = gcache->photonDNNEstimator->getSessions();
  }
}

std::unique_ptr<CacheData> GEDPhotonProducer::initializeGlobalCache(const edm::ParameterSet& config) {
  // this method is supposed to create, initialize and return a CacheData instance
  return std::make_unique<CacheData>(config);
}

void GEDPhotonProducer::endStream() {
  for (auto session : tfSessions_) {
    tensorflow::closeSession(session);
  }
}

void GEDPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& eventSetup) {
  using namespace edm;

  auto outputPhotonCollection_p = std::make_unique<reco::PhotonCollection>();
  edm::ValueMap<reco::PhotonRef> pfEGCandToPhotonMap;

  // Get the PhotonCore collection
  bool validPhotonCoreHandle = false;
  Handle<reco::PhotonCoreCollection> photonCoreHandle;
  bool validPhotonHandle = false;
  Handle<reco::PhotonCollection> photonHandle;
  //value maps for isolation
  edm::Handle<edm::ValueMap<float>> phoChargedIsolationMap;
  edm::Handle<edm::ValueMap<float>> phoNeutralHadronIsolationMap;
  edm::Handle<edm::ValueMap<float>> phoPhotonIsolationMap;
  edm::Handle<edm::ValueMap<float>> phoChargedWorstVtxIsoMap;
  edm::Handle<edm::ValueMap<float>> phoChargedWorstVtxGeomVetoIsoMap;
  edm::Handle<edm::ValueMap<float>> phoChargedPFPVIsoMap;

  edm::Handle<edm::ValueMap<float>> phoPFECALClusIsolationMap;
  edm::Handle<edm::ValueMap<float>> phoPFHCALClusIsolationMap;

  if (recoStep_.isFinal()) {
    theEvent.getByToken(photonProducerT_, photonHandle);
    //get isolation objects
    theEvent.getByToken(phoChargedIsolationToken_, phoChargedIsolationMap);
    theEvent.getByToken(phoNeutralHadronIsolationToken_, phoNeutralHadronIsolationMap);
    theEvent.getByToken(phoPhotonIsolationToken_, phoPhotonIsolationMap);
    theEvent.getByToken(phoChargedWorstVtxIsoToken_, phoChargedWorstVtxIsoMap);
    theEvent.getByToken(phoChargedWorstVtxGeomVetoIsoToken_, phoChargedWorstVtxGeomVetoIsoMap);
    theEvent.getByToken(phoChargedPFPVIsoToken_, phoChargedPFPVIsoMap);

    //OOT photons in legacy 80X re-miniAOD workflow dont have cluster isolation embed in them
    if (!phoPFECALClusIsolationToken_.isUninitialized()) {
      theEvent.getByToken(phoPFECALClusIsolationToken_, phoPFECALClusIsolationMap);
    }
    if (!phoPFHCALClusIsolationToken_.isUninitialized()) {
      theEvent.getByToken(phoPFHCALClusIsolationToken_, phoPFHCALClusIsolationMap);
    }

    if (photonHandle.isValid()) {
      validPhotonHandle = true;
    } else {
      throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the product " << photonProducer_.label() << "\n";
    }
  } else {
    theEvent.getByToken(photonCoreProducerT_, photonCoreHandle);
    if (photonCoreHandle.isValid()) {
      validPhotonCoreHandle = true;
    } else {
      throw cms::Exception("GEDPhotonProducer")
          << "Error! Can't get the photonCoreProducer " << photonProducer_.label() << "\n";
    }
  }

  // Get EcalRecHits
  auto const& barrelRecHits = theEvent.get(barrelEcalHits_);
  auto const& endcapRecHits = theEvent.get(endcapEcalHits_);
  auto const& preshowerRecHits = theEvent.get(preshowerHits_);

  Handle<reco::PFCandidateCollection> pfEGCandidateHandle;
  // Get the  PF refined cluster  collection
  if (not pfEgammaCandidates_.isUninitialized()) {
    theEvent.getByToken(pfEgammaCandidates_, pfEGCandidateHandle);
    if (!pfEGCandidateHandle.isValid()) {
      throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the pfEgammaCandidates";
    }
  }

  Handle<reco::PFCandidateCollection> pfCandidateHandle;

  if (recoStep_.isFinal()) {
    // Get the  PF candidates collection
    theEvent.getByToken(pfCandidates_, pfCandidateHandle);
    //OOT photons have no PF candidates so its not an error in this case
    if (!pfCandidateHandle.isValid() && !recoStep_.isOOT()) {
      throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the pfCandidates";
    }
  }

  // get the geometry from the event setup:
  caloGeom_ = &eventSetup.getData(caloGeometryToken_);

  // prepare access to hcal data
  if (hcalHelperCone_ != nullptr and hcalHelperBc_ != nullptr) {
    hcalHelperCone_->beginEvent(theEvent, eventSetup);
    hcalHelperBc_->beginEvent(theEvent, eventSetup);
  }

  auto const& topology = eventSetup.getData(caloTopologyToken_);
  auto const& thresholds = eventSetup.getData(ecalPFRechitThresholdsToken_);

  // Get the primary event vertex
  const reco::VertexCollection dummyVC;
  auto const& vertexCollection{usePrimaryVertex_ ? theEvent.get(vertexProducer_) : dummyVC};

  //  math::XYZPoint vtx(0.,0.,0.);
  //if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  // get the regression calculator ready
  photonEnergyCorrector_->init(eventSetup);
  if (photonEnergyCorrector_->gedRegression()) {
    photonEnergyCorrector_->gedRegression()->setEvent(theEvent);
    photonEnergyCorrector_->gedRegression()->setEventContent(eventSetup);
  }

  ///PF ECAL cluster based isolations
  ecalisoAlgo = std::make_unique<PhotonEcalPFClusterIsolation>(ecaldrMax_,
                                                               ecaldrVetoBarrel_,
                                                               ecaldrVetoEndcap_,
                                                               ecaletaStripBarrel_,
                                                               ecaletaStripEndcap_,
                                                               ecalenergyBarrel_,
                                                               ecalenergyEndcap_);

  hcalisoAlgo = std::make_unique<PhotonHcalPFClusterIsolation>(hcaldrMax_,
                                                               hcaldrVetoBarrel_,
                                                               hcaldrVetoEndcap_,
                                                               hcaletaStripBarrel_,
                                                               hcaletaStripEndcap_,
                                                               hcalenergyBarrel_,
                                                               hcalenergyEndcap_,
                                                               hcaluseEt_);

  int iSC = 0;  // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  if (validPhotonCoreHandle)
    fillPhotonCollection(theEvent,
                         eventSetup,
                         photonCoreHandle,
                         &topology,
                         &barrelRecHits,
                         &endcapRecHits,
                         &preshowerRecHits,
                         hcalHelperCone_.get(),
                         hcalHelperBc_.get(),
                         //vtx,
                         vertexCollection,
                         *outputPhotonCollection_p,
                         iSC,
                         thresholds);

  iSC = 0;
  if (validPhotonHandle && recoStep_.isFinal())
    fillPhotonCollection(theEvent,
                         eventSetup,
                         photonHandle,
                         pfCandidateHandle,
                         pfEGCandidateHandle,
                         theEvent.get(vertexProducer_),
                         *outputPhotonCollection_p,
                         iSC,
                         phoChargedIsolationMap,
                         phoNeutralHadronIsolationMap,
                         phoPhotonIsolationMap,
                         phoChargedWorstVtxIsoMap,
                         phoChargedWorstVtxGeomVetoIsoMap,
                         phoChargedPFPVIsoMap,
                         phoPFECALClusIsolationMap,
                         phoPFHCALClusIsolationMap);

  // put the product in the event
  edm::LogInfo("GEDPhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";

  // go back to run2-like 2 effective depths if desired - depth 1 is the normal depth 1, depth 2 is the sum over the rest
  if (hcalRun2EffDepth_) {
    for (auto& pho : *outputPhotonCollection_p)
      pho.hcalToRun2EffDepth();
  }
  const auto photonOrphHandle = theEvent.put(std::move(outputPhotonCollection_p), photonCollection_);

  if (!recoStep_.isFinal() && not pfEgammaCandidates_.isUninitialized()) {
    //// Define the value map which associate to each  Egamma-unbiassaed candidate (key-ref) the corresponding PhotonRef
    auto pfEGCandToPhotonMap_p = std::make_unique<edm::ValueMap<reco::PhotonRef>>();
    edm::ValueMap<reco::PhotonRef>::Filler filler(*pfEGCandToPhotonMap_p);
    unsigned nObj = pfEGCandidateHandle->size();
    std::vector<reco::PhotonRef> values(nObj);
    //// Fill the value map which associate to each Photon (key) the corresponding Egamma-unbiassaed candidate (value-ref)
    for (unsigned int lCand = 0; lCand < nObj; lCand++) {
      reco::PFCandidateRef pfCandRef(reco::PFCandidateRef(pfEGCandidateHandle, lCand));
      reco::SuperClusterRef pfScRef = pfCandRef->superClusterRef();

      for (unsigned int lSC = 0; lSC < photonOrphHandle->size(); lSC++) {
        reco::PhotonRef photonRef(reco::PhotonRef(photonOrphHandle, lSC));
        reco::SuperClusterRef scRef = photonRef->superCluster();
        if (pfScRef != scRef)
          continue;
        values[lCand] = photonRef;
      }
    }

    filler.insert(pfEGCandidateHandle, values.begin(), values.end());
    filler.fill();
    theEvent.put(std::move(pfEGCandToPhotonMap_p), valueMapPFCandPhoton_);
  }
}

void GEDPhotonProducer::fillPhotonCollection(edm::Event& evt,
                                             edm::EventSetup const& es,
                                             const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                                             const CaloTopology* topology,
                                             const EcalRecHitCollection* ecalBarrelHits,
                                             const EcalRecHitCollection* ecalEndcapHits,
                                             const EcalRecHitCollection* preshowerHits,
                                             const ElectronHcalHelper* hcalHelperCone,
                                             const ElectronHcalHelper* hcalHelperBc,
                                             const reco::VertexCollection& vertexCollection,
                                             reco::PhotonCollection& outputPhotonCollection,
                                             int& iSC,
                                             EcalPFRecHitThresholds const& thresholds) {
  const EcalRecHitCollection* hits = nullptr;
  std::vector<double> preselCutValues;
  std::vector<int> flags_, severitiesexcl_;

  for (unsigned int lSC = 0; lSC < photonCoreHandle->size(); lSC++) {
    reco::PhotonCoreRef coreRef(reco::PhotonCoreRef(photonCoreHandle, lSC));
    reco::SuperClusterRef parentSCRef = coreRef->parentSuperCluster();
    reco::SuperClusterRef scRef = coreRef->superCluster();

    //    const reco::SuperCluster* pClus=&(*scRef);
    iSC++;

    DetId::Detector thedet = scRef->seed()->hitsAndFractions()[0].first.det();
    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    if (subdet == EcalBarrel) {
      preselCutValues = preselCutValuesBarrel_;
      hits = ecalBarrelHits;
      flags_ = flagsexclEB_;
      severitiesexcl_ = severitiesexclEB_;
    } else if (subdet == EcalEndcap) {
      preselCutValues = preselCutValuesEndcap_;
      hits = ecalEndcapHits;
      flags_ = flagsexclEE_;
      severitiesexcl_ = severitiesexclEE_;
    } else if (EcalTools::isHGCalDet(thedet)) {
      preselCutValues = preselCutValuesEndcap_;
      hits = nullptr;
      flags_ = flagsexclEE_;
      severitiesexcl_ = severitiesexclEE_;
    } else {
      edm::LogWarning("") << "GEDPhotonProducer: do not know if it is a barrel or endcap SuperCluster: " << thedet
                          << ' ' << subdet;
    }

    // SC energy preselection
    if (parentSCRef.isNonnull() &&
        ptFast(parentSCRef->energy(), parentSCRef->position(), {0, 0, 0}) <= preselCutValues[0])
      continue;

    float maxXtal = (hits != nullptr ? EcalClusterTools::eMax(*(scRef->seed()), hits) : 0.f);

    //AA
    //Change these to consider severity level of hits
    float e1x5 = (hits != nullptr ? EcalClusterTools::e1x5(*(scRef->seed()), hits, topology) : 0.f);
    float e2x5 = (hits != nullptr ? EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    float e3x3 = (hits != nullptr ? EcalClusterTools::e3x3(*(scRef->seed()), hits, topology) : 0.f);
    float e5x5 = (hits != nullptr ? EcalClusterTools::e5x5(*(scRef->seed()), hits, topology) : 0.f);
    const auto& cov = (hits != nullptr ? EcalClusterTools::covariances(*(scRef->seed()), hits, topology, caloGeom_)
                                       : std::array<float, 3>({{0.f, 0.f, 0.f}}));
    // fractional local covariances
    const auto& locCov = (hits != nullptr ? EcalClusterTools::localCovariances(*(scRef->seed()), hits, topology)
                                          : std::array<float, 3>({{0.f, 0.f, 0.f}}));

    float sigmaEtaEta = std::sqrt(cov[0]);
    float sigmaIetaIeta = std::sqrt(locCov[0]);

    float full5x5_maxXtal = (hits != nullptr ? noZS::EcalClusterTools::eMax(*(scRef->seed()), hits) : 0.f);
    //AA
    //Change these to consider severity level of hits
    float full5x5_e1x5 = (hits != nullptr ? noZS::EcalClusterTools::e1x5(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e2x5 = (hits != nullptr ? noZS::EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e3x3 = (hits != nullptr ? noZS::EcalClusterTools::e3x3(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e5x5 = (hits != nullptr ? noZS::EcalClusterTools::e5x5(*(scRef->seed()), hits, topology) : 0.f);
    const auto& full5x5_cov =
        (hits != nullptr ? noZS::EcalClusterTools::covariances(*(scRef->seed()), hits, topology, caloGeom_)
                         : std::array<float, 3>({{0.f, 0.f, 0.f}}));
    // for full5x5 local covariances, do noise-cleaning
    // by passing per crystal PF recHit thresholds and mult values.
    // mult values for EB and EE were obtained by dedicated studies.
    const auto& full5x5_locCov =
        (hits != nullptr ? noZS::EcalClusterTools::localCovariances(*(scRef->seed()),
                                                                    hits,
                                                                    topology,
                                                                    EgammaLocalCovParamDefaults::kRelEnCut,
                                                                    &thresholds,
                                                                    multThresEB_,
                                                                    multThresEE_)
                         : std::array<float, 3>({{0.f, 0.f, 0.f}}));

    float full5x5_sigmaEtaEta = sqrt(full5x5_cov[0]);
    float full5x5_sigmaIetaIeta = sqrt(full5x5_locCov[0]);
    float full5x5_sigmaIetaIphi = full5x5_locCov[1];

    // compute position of ECAL shower
    math::XYZPoint caloPosition = scRef->position();

    //// energy determination -- Default to create the candidate. Afterwards corrections are applied
    double photonEnergy = 1.;
    math::XYZPoint vtx(0., 0., 0.);
    if (!vertexCollection.empty())
      vtx = vertexCollection.begin()->position();
    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    //math::XYZVector momentum = direction.unit() * photonEnergy ;
    math::XYZVector momentum = direction.unit();

    // Create dummy candidate with unit momentum and zero energy to allow setting of all variables. The energy is set for last.
    math::XYZTLorentzVectorD p4(momentum.x(), momentum.y(), momentum.z(), photonEnergy);
    reco::Photon newCandidate(p4, caloPosition, coreRef, vtx);

    //std::cout << " standard p4 before " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;
    //std::cout << " type " <<newCandidate.getCandidateP4type() <<  " standard p4 after " << newCandidate.p4() << " energy " << newCandidate.energy() << std::endl;

    // Calculate fiducial flags and isolation variable. Blocked are filled from the isolationCalculator
    reco::Photon::FiducialFlags fiducialFlags;
    reco::Photon::IsolationVariables isolVarR03, isolVarR04;
    if (!EcalTools::isHGCalDet(thedet)) {
      photonIsoCalculator_->calculate(&newCandidate, evt, es, fiducialFlags, isolVarR04, isolVarR03);
    }
    newCandidate.setFiducialVolumeFlags(fiducialFlags);
    newCandidate.setIsolationVariables(isolVarR04, isolVarR03);

    /// fill shower shape block
    reco::Photon::ShowerShape showerShape;
    showerShape.e1x5 = e1x5;
    showerShape.e2x5 = e2x5;
    showerShape.e3x3 = e3x3;
    showerShape.e5x5 = e5x5;
    showerShape.maxEnergyXtal = maxXtal;
    showerShape.sigmaEtaEta = sigmaEtaEta;
    showerShape.sigmaIetaIeta = sigmaIetaIeta;
    for (uint id = 0; id < showerShape.hcalOverEcal.size(); ++id) {
      showerShape.hcalOverEcal[id] =
          (hcalHelperCone != nullptr) ? hcalHelperCone->hcalESum(*scRef, id + 1) / scRef->energy() : 0.f;

      showerShape.hcalOverEcalBc[id] =
          (hcalHelperBc != nullptr) ? hcalHelperBc->hcalESum(*scRef, id + 1) / scRef->energy() : 0.f;
    }
    showerShape.invalidHcal = (hcalHelperBc != nullptr) ? !hcalHelperBc->hasActiveHcal(*scRef) : false;
    if (hcalHelperBc != nullptr)
      showerShape.hcalTowersBehindClusters = hcalHelperBc->hcalTowersBehindClusters(*scRef);
    showerShape.pre7DepthHcal = false;

    /// fill extra shower shapes
    const float spp = (!edm::isFinite(locCov[2]) ? 0. : sqrt(locCov[2]));
    const float sep = locCov[1];
    showerShape.sigmaIetaIphi = sep;
    showerShape.sigmaIphiIphi = spp;
    showerShape.e2nd = (hits != nullptr ? EcalClusterTools::e2nd(*(scRef->seed()), hits) : 0.f);
    showerShape.eTop = (hits != nullptr ? EcalClusterTools::eTop(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.eLeft = (hits != nullptr ? EcalClusterTools::eLeft(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.eRight = (hits != nullptr ? EcalClusterTools::eRight(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.eBottom = (hits != nullptr ? EcalClusterTools::eBottom(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e1x3 = (hits != nullptr ? EcalClusterTools::e1x3(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x2 = (hits != nullptr ? EcalClusterTools::e2x2(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x5Max = (hits != nullptr ? EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x5Left = (hits != nullptr ? EcalClusterTools::e2x5Left(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x5Right = (hits != nullptr ? EcalClusterTools::e2x5Right(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x5Top = (hits != nullptr ? EcalClusterTools::e2x5Top(*(scRef->seed()), hits, topology) : 0.f);
    showerShape.e2x5Bottom = (hits != nullptr ? EcalClusterTools::e2x5Bottom(*(scRef->seed()), hits, topology) : 0.f);
    if (hits) {
      Cluster2ndMoments clus2ndMoments = EcalClusterTools::cluster2ndMoments(*(scRef->seed()), *hits);
      showerShape.smMajor = clus2ndMoments.sMaj;
      showerShape.smMinor = clus2ndMoments.sMin;
      showerShape.smAlpha = clus2ndMoments.alpha;
    } else {
      showerShape.smMajor = 0.f;
      showerShape.smMinor = 0.f;
      showerShape.smAlpha = 0.f;
    }

    // fill preshower shapes
    EcalClusterLazyTools toolsforES(
        evt, ecalClusterESGetTokens_.get(es), barrelEcalHits_, endcapEcalHits_, preshowerHits_);
    const float sigmaRR = toolsforES.eseffsirir(*scRef);
    showerShape.effSigmaRR = sigmaRR;
    newCandidate.setShowerShapeVariables(showerShape);

    const reco::CaloCluster& seedCluster = *(scRef->seed());
    DetId seedXtalId = seedCluster.seed();
    int nSaturatedXtals = 0;
    bool isSeedSaturated = false;
    if (hits != nullptr) {
      const auto hitsAndFractions = scRef->hitsAndFractions();
      for (auto const& hitFractionPair : hitsAndFractions) {
        auto&& ecalRecHit = hits->find(hitFractionPair.first);
        if (ecalRecHit == hits->end())
          continue;
        if (ecalRecHit->checkFlag(EcalRecHit::Flags::kSaturated)) {
          nSaturatedXtals++;
          if (seedXtalId == ecalRecHit->detid())
            isSeedSaturated = true;
        }
      }
    }
    reco::Photon::SaturationInfo saturationInfo;
    saturationInfo.nSaturatedXtals = nSaturatedXtals;
    saturationInfo.isSeedSaturated = isSeedSaturated;
    newCandidate.setSaturationInfo(saturationInfo);

    /// fill full5x5 shower shape block
    reco::Photon::ShowerShape full5x5_showerShape;
    full5x5_showerShape.e1x5 = full5x5_e1x5;
    full5x5_showerShape.e2x5 = full5x5_e2x5;
    full5x5_showerShape.e3x3 = full5x5_e3x3;
    full5x5_showerShape.e5x5 = full5x5_e5x5;
    full5x5_showerShape.maxEnergyXtal = full5x5_maxXtal;
    full5x5_showerShape.sigmaEtaEta = full5x5_sigmaEtaEta;
    full5x5_showerShape.sigmaIetaIeta = full5x5_sigmaIetaIeta;
    /// fill extra full5x5 shower shapes
    const float full5x5_spp = (!edm::isFinite(full5x5_locCov[2]) ? 0. : std::sqrt(full5x5_locCov[2]));
    const float full5x5_sep = full5x5_sigmaIetaIphi;
    full5x5_showerShape.sigmaIetaIphi = full5x5_sep;
    full5x5_showerShape.sigmaIphiIphi = full5x5_spp;
    full5x5_showerShape.e2nd = (hits != nullptr ? noZS::EcalClusterTools::e2nd(*(scRef->seed()), hits) : 0.f);
    full5x5_showerShape.eTop = (hits != nullptr ? noZS::EcalClusterTools::eTop(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.eLeft =
        (hits != nullptr ? noZS::EcalClusterTools::eLeft(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.eRight =
        (hits != nullptr ? noZS::EcalClusterTools::eRight(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.eBottom =
        (hits != nullptr ? noZS::EcalClusterTools::eBottom(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e1x3 = (hits != nullptr ? noZS::EcalClusterTools::e1x3(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x2 = (hits != nullptr ? noZS::EcalClusterTools::e2x2(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x5Max =
        (hits != nullptr ? noZS::EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x5Left =
        (hits != nullptr ? noZS::EcalClusterTools::e2x5Left(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x5Right =
        (hits != nullptr ? noZS::EcalClusterTools::e2x5Right(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x5Top =
        (hits != nullptr ? noZS::EcalClusterTools::e2x5Top(*(scRef->seed()), hits, topology) : 0.f);
    full5x5_showerShape.e2x5Bottom =
        (hits != nullptr ? noZS::EcalClusterTools::e2x5Bottom(*(scRef->seed()), hits, topology) : 0.f);
    if (hits) {
      Cluster2ndMoments clus2ndMoments = noZS::EcalClusterTools::cluster2ndMoments(*(scRef->seed()), *hits);
      full5x5_showerShape.smMajor = clus2ndMoments.sMaj;
      full5x5_showerShape.smMinor = clus2ndMoments.sMin;
      full5x5_showerShape.smAlpha = clus2ndMoments.alpha;
    } else {
      full5x5_showerShape.smMajor = 0.f;
      full5x5_showerShape.smMinor = 0.f;
      full5x5_showerShape.smAlpha = 0.f;
    }
    // fill preshower shapes
    full5x5_showerShape.effSigmaRR = sigmaRR;
    for (uint id = 0; id < full5x5_showerShape.hcalOverEcal.size(); ++id) {
      full5x5_showerShape.hcalOverEcal[id] =
          (hcalHelperCone != nullptr) ? hcalHelperCone->hcalESum(*scRef, id + 1) / full5x5_e5x5 : 0.f;
      full5x5_showerShape.hcalOverEcalBc[id] =
          (hcalHelperBc != nullptr) ? hcalHelperBc->hcalESum(*scRef, id + 1) / full5x5_e5x5 : 0.f;
    }
    full5x5_showerShape.pre7DepthHcal = false;
    newCandidate.full5x5_setShowerShapeVariables(full5x5_showerShape);

    //get the pointer for the photon object
    edm::Ptr<reco::PhotonCore> photonPtr(photonCoreHandle, lSC);

    // New in CMSSW_12_1_0 for PFID with DNNs
    // The PFIso values are computed in the first loop on gedPhotonsTmp to make them available as DNN inputs.
    // They are computed with the same inputs and algo as the final PFiso variables computed in the second loop after PF.
    // Get PFClusters for PFID only if the PFID DNN evaluation is enabled
    if (dnnPFidEnabled_) {
      auto clusterHandle = evt.getHandle(pfClusterProducer_);
      std::vector<edm::Handle<reco::PFClusterCollection>> clusterHandles{evt.getHandle(pfClusterProducerHCAL_)};
      if (useHF_) {
        clusterHandles.push_back(evt.getHandle(pfClusterProducerHFEM_));
        clusterHandles.push_back(evt.getHandle(pfClusterProducerHFHAD_));
      }
      reco::Photon::PflowIsolationVariables pfIso;
      pfIso.sumEcalClusterEt = ecalisoAlgo->getSum(newCandidate, clusterHandle);
      pfIso.sumHcalClusterEt = hcalisoAlgo->getSum(newCandidate, clusterHandles);

      newCandidate.setPflowIsolationVariables(pfIso);
    }

    /// get ecal photon specific corrected energy
    /// plus values from regressions     and store them in the Photon
    // Photon candidate takes by default (set in photons_cfi.py)
    // a 4-momentum derived from the ecal photon-specific corrections.
    if (!EcalTools::isHGCalDet(thedet)) {
      photonEnergyCorrector_->calculate(evt, newCandidate, subdet, vertexCollection, es);
      if (candidateP4type_ == "fromEcalEnergy") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::ecal_photons));
        newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
        newCandidate.setMass(0.0);
      } else if (candidateP4type_ == "fromRegression1") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression1));
        newCandidate.setCandidateP4type(reco::Photon::regression1);
        newCandidate.setMass(0.0);
      } else if (candidateP4type_ == "fromRegression2") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
        newCandidate.setCandidateP4type(reco::Photon::regression2);
        newCandidate.setMass(0.0);
      } else if (candidateP4type_ == "fromRefinedSCRegression") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
        newCandidate.setCandidateP4type(reco::Photon::regression2);
        newCandidate.setMass(0.0);
      }
    } else {
      math::XYZVector gamma_momentum = direction.unit() * scRef->energy();
      math::PtEtaPhiMLorentzVector p4(gamma_momentum.rho(), gamma_momentum.eta(), gamma_momentum.phi(), 0.0);
      newCandidate.setP4(p4);
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
      // Make it an EE photon
      reco::Photon::FiducialFlags fiducialFlags;
      fiducialFlags.isEE = true;
      newCandidate.setFiducialVolumeFlags(fiducialFlags);
    }

    // fill MIP Vairables for Halo: Block for MIP are filled from PhotonMIPHaloTagger
    reco::Photon::MIPVariables mipVar;
    if (subdet == EcalBarrel && runMIPTagger_) {
      photonMIPHaloTagger_->MIPcalculate(&newCandidate, evt, es, mipVar);
      newCandidate.setMIPVariables(mipVar);
    }

    /// Pre-selection loose  isolation cuts
    bool isLooseEM = true;
    if (newCandidate.pt() < highEt_) {
      if (newCandidate.hadronicOverEm() >= preselCutValues[1])
        isLooseEM = false;
      if (newCandidate.ecalRecHitSumEtConeDR04() > preselCutValues[2] + preselCutValues[3] * newCandidate.pt())
        isLooseEM = false;
      if (newCandidate.hcalTowerSumEtConeDR04() > preselCutValues[4] + preselCutValues[5] * newCandidate.pt())
        isLooseEM = false;
      if (newCandidate.nTrkSolidConeDR04() > int(preselCutValues[6]))
        isLooseEM = false;
      if (newCandidate.nTrkHollowConeDR04() > int(preselCutValues[7]))
        isLooseEM = false;
      if (newCandidate.trkSumPtSolidConeDR04() > preselCutValues[8])
        isLooseEM = false;
      if (newCandidate.trkSumPtHollowConeDR04() > preselCutValues[9])
        isLooseEM = false;
      if (newCandidate.sigmaIetaIeta() > preselCutValues[10])
        isLooseEM = false;
    }

    if (isLooseEM)
      outputPhotonCollection.push_back(newCandidate);
  }

  if (dnnPFidEnabled_) {
    // Here send the list of photons to the PhotonDNNEstimator and get back the values for all the photons in one go
    LogDebug("GEDPhotonProducer") << "Getting DNN PFId for photons";
    const auto& dnn_photon_pfid = globalCache()->photonDNNEstimator->evaluate(outputPhotonCollection, tfSessions_);
    size_t ipho = 0;
    for (auto& photon : outputPhotonCollection) {
      const auto& [iModel, values] = dnn_photon_pfid[ipho];
      reco::Photon::PflowIDVariables pfID;
      // The model index it is not useful for the moment
      pfID.dnn = values[0];
      photon.setPflowIDVariables(pfID);
      ipho++;
    }
  }
}

void GEDPhotonProducer::fillPhotonCollection(edm::Event& evt,
                                             edm::EventSetup const& es,
                                             const edm::Handle<reco::PhotonCollection>& photonHandle,
                                             const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
                                             const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
                                             reco::VertexCollection const& vertexCollection,
                                             reco::PhotonCollection& outputPhotonCollection,
                                             int& iSC,
                                             const edm::Handle<edm::ValueMap<float>>& chargedHadrons,
                                             const edm::Handle<edm::ValueMap<float>>& neutralHadrons,
                                             const edm::Handle<edm::ValueMap<float>>& photons,
                                             const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtx,
                                             const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtxGeomVeto,
                                             const edm::Handle<edm::ValueMap<float>>& chargedHadronsPFPV,
                                             const edm::Handle<edm::ValueMap<float>>& pfEcalClusters,
                                             const edm::Handle<edm::ValueMap<float>>& pfHcalClusters) {
  std::vector<double> preselCutValues;

  for (unsigned int lSC = 0; lSC < photonHandle->size(); lSC++) {
    reco::PhotonRef phoRef(reco::PhotonRef(photonHandle, lSC));
    reco::SuperClusterRef parentSCRef = phoRef->parentSuperCluster();
    reco::SuperClusterRef scRef = phoRef->superCluster();
    DetId::Detector thedet = scRef->seed()->hitsAndFractions()[0].first.det();
    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    if (subdet == EcalBarrel) {
      preselCutValues = preselCutValuesBarrel_;
    } else if (subdet == EcalEndcap) {
      preselCutValues = preselCutValuesEndcap_;
    } else if (EcalTools::isHGCalDet(thedet)) {
      preselCutValues = preselCutValuesEndcap_;
    } else {
      edm::LogWarning("") << "GEDPhotonProducer: do not know if it is a barrel or endcap SuperCluster" << thedet << ' '
                          << subdet;
    }

    // SC energy preselection
    if (parentSCRef.isNonnull() &&
        ptFast(parentSCRef->energy(), parentSCRef->position(), {0, 0, 0}) <= preselCutValues[0])
      continue;

    reco::Photon newCandidate(*phoRef);
    iSC++;

    if (runMVABasedHaloTagger_) {  ///sets values only for EE, for EB it always returns 1
      float BHmva = photonMVABasedHaloTagger_->calculateMVA(&newCandidate, globalCache()->haloTaggerGBR.get(), evt, es);
      newCandidate.setHaloTaggerMVAVal(BHmva);
    }

    // Calculate the PF isolation
    reco::Photon::PflowIsolationVariables pfIso;
    // The PFID are not recomputed since they have been already computed in the first loop with the DNN

    //get the pointer for the photon object
    edm::Ptr<reco::Photon> photonPtr(photonHandle, lSC);

    if (!recoStep_.isOOT()) {  //out of time photons do not have PF info so skip in this case
      pfIso.chargedHadronIso = (*chargedHadrons)[photonPtr];
      pfIso.neutralHadronIso = (*neutralHadrons)[photonPtr];
      pfIso.photonIso = (*photons)[photonPtr];
      pfIso.chargedHadronWorstVtxIso = (*chargedHadronsWorstVtx)[photonPtr];
      pfIso.chargedHadronWorstVtxGeomVetoIso = (*chargedHadronsWorstVtxGeomVeto)[photonPtr];
      pfIso.chargedHadronPFPVIso = (*chargedHadronsPFPV)[photonPtr];
    }

    //OOT photons in legacy 80X reminiAOD workflow dont have pf cluster isolation embeded into them at this stage
    // They have been already computed in the first loop on gedPhotonsTmp but better to compute them again here.
    pfIso.sumEcalClusterEt = !phoPFECALClusIsolationToken_.isUninitialized() ? (*pfEcalClusters)[photonPtr] : 0.;
    pfIso.sumHcalClusterEt = !phoPFHCALClusIsolationToken_.isUninitialized() ? (*pfHcalClusters)[photonPtr] : 0.;
    newCandidate.setPflowIsolationVariables(pfIso);

    // do the regression
    photonEnergyCorrector_->calculate(evt, newCandidate, subdet, vertexCollection, es);
    if (candidateP4type_ == "fromEcalEnergy") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::ecal_photons));
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
      newCandidate.setMass(0.0);
    } else if (candidateP4type_ == "fromRegression1") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression1));
      newCandidate.setCandidateP4type(reco::Photon::regression1);
      newCandidate.setMass(0.0);
    } else if (candidateP4type_ == "fromRegression2") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
      newCandidate.setCandidateP4type(reco::Photon::regression2);
      newCandidate.setMass(0.0);
    } else if (candidateP4type_ == "fromRefinedSCRegression") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
      newCandidate.setCandidateP4type(reco::Photon::regression2);
      newCandidate.setMass(0.0);
    }

    outputPhotonCollection.push_back(newCandidate);
  }
}
