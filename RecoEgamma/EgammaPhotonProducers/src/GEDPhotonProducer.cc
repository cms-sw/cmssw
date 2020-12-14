/** \class GEDPhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
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
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

class GEDPhotonProducer : public edm::stream::EDProducer<> {
public:
  GEDPhotonProducer(const edm::ParameterSet& ps);

  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const&, edm::EventSetup const&) final {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

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
                            CaloTowerCollection const* hcalTowers,
                            const reco::VertexCollection& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC);

  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCollection>& photonHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
                            edm::Handle<reco::VertexCollection>& pvVertices,
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
  edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
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

  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double highEt_;
  double minR9Barrel_;
  double minR9Endcap_;
  bool runMIPTagger_;

  RecoStepInfo recoStep_;

  bool usePrimaryVertex_;

  CaloGeometry const* caloGeom_ = nullptr;

  //MIP
  std::unique_ptr<PhotonMIPHaloTagger> photonMIPHaloTagger_ = nullptr;

  std::vector<double> preselCutValuesBarrel_;
  std::vector<double> preselCutValuesEndcap_;

  std::unique_ptr<PhotonEnergyCorrector> photonEnergyCorrector_ = nullptr;
  std::string candidateP4type_;

  bool checkHcalStatus_;

  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
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

GEDPhotonProducer::GEDPhotonProducer(const edm::ParameterSet& config)
    : photonProducer_{config.getParameter<edm::InputTag>("photonProducer")},
      ecalClusterESGetTokens_{consumesCollector()},
      recoStep_(config.getParameter<std::string>("reconstructionStep")),
      caloTopologyToken_{esConsumes()},
      caloGeometryToken_{esConsumes()} {
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

  auto hcTow = config.getParameter<edm::InputTag>("hcalTowers");
  if (not hcTow.label().empty()) {
    hcalTowers_ = consumes(hcTow);
  }
  //
  photonCollection_ = config.getParameter<std::string>("outputPhotonCollection");
  hOverEConeSize_ = config.getParameter<double>("hOverEConeSize");
  highEt_ = config.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_ = config.getParameter<double>("minR9Barrel");
  minR9Endcap_ = config.getParameter<double>("minR9Endcap");
  usePrimaryVertex_ = config.getParameter<bool>("usePrimaryVertex");
  runMIPTagger_ = config.getParameter<bool>("runMIPTagger");

  candidateP4type_ = config.getParameter<std::string>("candidateP4type");
  valueMapPFCandPhoton_ = config.getParameter<std::string>("valueMapPhotons");

  //AA
  //Flags and Severities to be excluded from photon calculations
  const auto flagnamesEB = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");

  const auto flagnamesEE = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const auto severitynamesEB = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const auto severitynamesEE = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  photonEnergyCorrector_ = std::make_unique<PhotonEnergyCorrector>(config, consumesCollector());

  //AA

  //

  // Parameters for the position calculation:
  //  std::map<std::string,double> providedParameters;
  // providedParameters.insert(std::make_pair("LogWeighted",config.getParameter<bool>("posCalc_logweight")));
  //providedParameters.insert(std::make_pair("T0_barl",config.getParameter<double>("posCalc_t0_barl")));
  //providedParameters.insert(std::make_pair("T0_endc",config.getParameter<double>("posCalc_t0_endc")));
  //providedParameters.insert(std::make_pair("T0_endcPresh",config.getParameter<double>("posCalc_t0_endcPresh")));
  //providedParameters.insert(std::make_pair("W0",config.getParameter<double>("posCalc_w0")));
  //providedParameters.insert(std::make_pair("X0",config.getParameter<double>("posCalc_x0")));
  //posCalculator_ = PositionCalc(providedParameters);
  // cut values for pre-selection
  preselCutValuesBarrel_ = {config.getParameter<double>("minSCEtBarrel"),
                            config.getParameter<double>("maxHoverEBarrel"),
                            config.getParameter<double>("ecalRecHitSumEtOffsetBarrel"),
                            config.getParameter<double>("ecalRecHitSumEtSlopeBarrel"),
                            config.getParameter<double>("hcalTowerSumEtOffsetBarrel"),
                            config.getParameter<double>("hcalTowerSumEtSlopeBarrel"),
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
                            config.getParameter<double>("hcalTowerSumEtOffsetEndcap"),
                            config.getParameter<double>("hcalTowerSumEtSlopeEndcap"),
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

  checkHcalStatus_ = config.getParameter<bool>("checkHcalStatus");

  // Register the product
  produces<reco::PhotonCollection>(photonCollection_);
  if (not pfEgammaCandidates_.isUninitialized()) {
    produces<edm::ValueMap<reco::PhotonRef>>(valueMapPFCandPhoton_);
  }
}

void GEDPhotonProducer::beginRun(edm::Run const& r, edm::EventSetup const& eventSetup) {
  if (!recoStep_.isFinal()) {
    photonEnergyCorrector_->init(eventSetup);
  }
}

void GEDPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& eventSetup) {
  using namespace edm;
  //  nEvt_++;

  reco::PhotonCollection outputPhotonCollection;
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
  bool validEcalRecHits = true;
  const EcalRecHitCollection dummyEB;
  auto barrelHitHandle = theEvent.getHandle(barrelEcalHits_);
  if (!barrelHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the barrelEcalHits";
  }
  const EcalRecHitCollection& barrelRecHits(validEcalRecHits ? *(barrelHitHandle.product()) : dummyEB);

  auto endcapHitHandle = theEvent.getHandle(endcapEcalHits_);
  const EcalRecHitCollection dummyEE;
  if (!endcapHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the endcapEcalHits";
  }
  const EcalRecHitCollection& endcapRecHits(validEcalRecHits ? *(endcapHitHandle.product()) : dummyEE);

  bool validPreshowerRecHits = true;
  auto preshowerHitHandle = theEvent.getHandle(preshowerHits_);
  EcalRecHitCollection preshowerRecHits;
  if (!preshowerHitHandle.isValid()) {
    throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the preshowerEcalHits";
  }
  if (validPreshowerRecHits) {
    preshowerRecHits = *preshowerHitHandle;
  }

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

  // get Hcal towers collection
  CaloTowerCollection const* hcalTowers = hcalTowers_.isUninitialized() ? nullptr : &theEvent.get(hcalTowers_);

  // get the geometry from the event setup:
  caloGeom_ = &eventSetup.getData(caloGeometryToken_);

  auto const& topology = eventSetup.getData(caloTopologyToken_);

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  const reco::VertexCollection dummyVC;
  bool validVertex = true;
  if (usePrimaryVertex_) {
    theEvent.getByToken(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      throw cms::Exception("GEDPhotonProducer") << "Error! Can't get the product primary Vertex Collection";
    }
  }
  auto const& vertexCollection{usePrimaryVertex_ && validVertex ? *vertexHandle : dummyVC};

  //  math::XYZPoint vtx(0.,0.,0.);
  //if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  // get the regression calculator ready
  photonEnergyCorrector_->init(eventSetup);
  if (photonEnergyCorrector_->gedRegression()) {
    photonEnergyCorrector_->gedRegression()->setEvent(theEvent);
    photonEnergyCorrector_->gedRegression()->setEventContent(eventSetup);
  }

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
                         hcalTowers,
                         //vtx,
                         vertexCollection,
                         outputPhotonCollection,
                         iSC);

  iSC = 0;
  if (validPhotonHandle && recoStep_.isFinal())
    fillPhotonCollection(theEvent,
                         eventSetup,
                         photonHandle,
                         pfCandidateHandle,
                         pfEGCandidateHandle,
                         vertexHandle,
                         outputPhotonCollection,
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
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(), outputPhotonCollection.end());
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
                                             CaloTowerCollection const* hcalTowers,
                                             const reco::VertexCollection& vertexCollection,
                                             reco::PhotonCollection& outputPhotonCollection,
                                             int& iSC) {
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
    // calculate HoE

    double HoE1, HoE2;
    HoE1 = HoE2 = 0.;

    std::vector<CaloTowerDetId> TowersBehindClus;
    float hcalDepth1OverEcalBc, hcalDepth2OverEcalBc;
    hcalDepth1OverEcalBc = hcalDepth2OverEcalBc = 0.f;
    bool invalidHcal = false;

    if (not hcalTowers_.isUninitialized()) {
      EgammaTowerIsolation towerIso1(hOverEConeSize_, 0., 0., 1, hcalTowers);
      EgammaTowerIsolation towerIso2(hOverEConeSize_, 0., 0., 2, hcalTowers);
      HoE1 = towerIso1.getTowerESum(&(*scRef)) / scRef->energy();
      HoE2 = towerIso2.getTowerESum(&(*scRef)) / scRef->energy();

      edm::ESHandle<CaloTowerConstituentsMap> ctmaph;
      es.get<CaloGeometryRecord>().get(ctmaph);

      edm::ESHandle<HcalChannelQuality> hcalQuality;
      es.get<HcalChannelQualityRcd>().get("withTopo", hcalQuality);

      edm::ESHandle<HcalTopology> hcalTopology;
      es.get<HcalRecNumberingRecord>().get(hcalTopology);

      TowersBehindClus = egamma::towersOf(*scRef, *ctmaph);
      hcalDepth1OverEcalBc = egamma::depth1HcalESum(TowersBehindClus, *hcalTowers) / scRef->energy();
      hcalDepth2OverEcalBc = egamma::depth2HcalESum(TowersBehindClus, *hcalTowers) / scRef->energy();

      if (checkHcalStatus_ && hcalDepth1OverEcalBc == 0 && hcalDepth2OverEcalBc == 0) {
        invalidHcal = !egamma::hasActiveHcal(TowersBehindClus, *ctmaph, *hcalQuality, *hcalTopology);
      }
    }

    //    std::cout << " GEDPhotonProducer calculation of HoE with towers in a cone " << HoE1  << "  " << HoE2 << std::endl;
    //std::cout << " GEDPhotonProducer calcualtion of HoE with towers behind the BCs " << hcalDepth1OverEcalBc  << "  " << hcalDepth2OverEcalBc << std::endl;

    float maxXtal = (hits != nullptr ? EcalClusterTools::eMax(*(scRef->seed()), hits) : 0.f);
    //AA
    //Change these to consider severity level of hits
    float e1x5 = (hits != nullptr ? EcalClusterTools::e1x5(*(scRef->seed()), hits, topology) : 0.f);
    float e2x5 = (hits != nullptr ? EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    float e3x3 = (hits != nullptr ? EcalClusterTools::e3x3(*(scRef->seed()), hits, topology) : 0.f);
    float e5x5 = (hits != nullptr ? EcalClusterTools::e5x5(*(scRef->seed()), hits, topology) : 0.f);
    std::vector<float> cov =
        (hits != nullptr ? EcalClusterTools::covariances(*(scRef->seed()), hits, topology, caloGeom_)
                         : std::vector<float>({0.f, 0.f, 0.f}));
    std::vector<float> locCov = (hits != nullptr ? EcalClusterTools::localCovariances(*(scRef->seed()), hits, topology)
                                                 : std::vector<float>({0.f, 0.f, 0.f}));

    float sigmaEtaEta = std::sqrt(cov[0]);
    float sigmaIetaIeta = std::sqrt(locCov[0]);

    float full5x5_maxXtal = (hits != nullptr ? noZS::EcalClusterTools::eMax(*(scRef->seed()), hits) : 0.f);
    //AA
    //Change these to consider severity level of hits
    float full5x5_e1x5 = (hits != nullptr ? noZS::EcalClusterTools::e1x5(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e2x5 = (hits != nullptr ? noZS::EcalClusterTools::e2x5Max(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e3x3 = (hits != nullptr ? noZS::EcalClusterTools::e3x3(*(scRef->seed()), hits, topology) : 0.f);
    float full5x5_e5x5 = (hits != nullptr ? noZS::EcalClusterTools::e5x5(*(scRef->seed()), hits, topology) : 0.f);
    std::vector<float> full5x5_cov =
        (hits != nullptr ? noZS::EcalClusterTools::covariances(*(scRef->seed()), hits, topology, caloGeom_)
                         : std::vector<float>({0.f, 0.f, 0.f}));
    std::vector<float> full5x5_locCov =
        (hits != nullptr ? noZS::EcalClusterTools::localCovariances(*(scRef->seed()), hits, topology)
                         : std::vector<float>({0.f, 0.f, 0.f}));

    float full5x5_sigmaEtaEta = sqrt(full5x5_cov[0]);
    float full5x5_sigmaIetaIeta = sqrt(full5x5_locCov[0]);

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
    showerShape.hcalDepth1OverEcal = HoE1;
    showerShape.hcalDepth2OverEcal = HoE2;
    showerShape.hcalDepth1OverEcalBc = hcalDepth1OverEcalBc;
    showerShape.hcalDepth2OverEcalBc = hcalDepth2OverEcalBc;
    showerShape.hcalTowersBehindClusters = TowersBehindClus;
    showerShape.invalidHcal = invalidHcal;
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
    const float full5x5_sep = full5x5_locCov[1];
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
    newCandidate.full5x5_setShowerShapeVariables(full5x5_showerShape);

    /// get ecal photon specific corrected energy
    /// plus values from regressions     and store them in the Photon
    // Photon candidate takes by default (set in photons_cfi.py)
    // a 4-momentum derived from the ecal photon-specific corrections.
    if (!EcalTools::isHGCalDet(thedet)) {
      photonEnergyCorrector_->calculate(evt, newCandidate, subdet, vertexCollection, es);
      if (candidateP4type_ == "fromEcalEnergy") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::ecal_photons));
        newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
      } else if (candidateP4type_ == "fromRegression1") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression1));
        newCandidate.setCandidateP4type(reco::Photon::regression1);
      } else if (candidateP4type_ == "fromRegression2") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
        newCandidate.setCandidateP4type(reco::Photon::regression2);
      } else if (candidateP4type_ == "fromRefinedSCRegression") {
        newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
        newCandidate.setCandidateP4type(reco::Photon::regression2);
      }
    } else {
      math::XYZVector gamma_momentum = direction.unit() * scRef->energy();
      math::XYZTLorentzVectorD p4(gamma_momentum.x(), gamma_momentum.y(), gamma_momentum.z(), scRef->energy());
      newCandidate.setP4(p4);
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
      // Make it an EE photon
      reco::Photon::FiducialFlags fiducialFlags;
      fiducialFlags.isEE = true;
      newCandidate.setFiducialVolumeFlags(fiducialFlags);
    }

    //       std::cout << " final p4 " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;

    // std::cout << " GEDPhotonProducer from candidate HoE with towers in a cone " << newCandidate.hadronicOverEm()  << "  " <<  newCandidate.hadronicDepth1OverEm()  << " " <<  newCandidate.hadronicDepth2OverEm()  << std::endl;
    //    std::cout << " GEDPhotonProducer from candidate  of HoE with towers behind the BCs " <<  newCandidate.hadTowOverEm()  << "  " << newCandidate.hadTowDepth1OverEm() << " " << newCandidate.hadTowDepth2OverEm() << std::endl;

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
}

void GEDPhotonProducer::fillPhotonCollection(edm::Event& evt,
                                             edm::EventSetup const& es,
                                             const edm::Handle<reco::PhotonCollection>& photonHandle,
                                             const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
                                             const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
                                             edm::Handle<reco::VertexCollection>& vertexHandle,
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

    // Calculate the PF isolation and ID - for the time being there is no calculation. Only the setting
    reco::Photon::PflowIsolationVariables pfIso;
    reco::Photon::PflowIDVariables pfID;

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
    pfIso.sumEcalClusterEt = !phoPFECALClusIsolationToken_.isUninitialized() ? (*pfEcalClusters)[photonPtr] : 0.;
    pfIso.sumHcalClusterEt = !phoPFHCALClusIsolationToken_.isUninitialized() ? (*pfHcalClusters)[photonPtr] : 0.;

    newCandidate.setPflowIsolationVariables(pfIso);
    newCandidate.setPflowIDVariables(pfID);

    // do the regression
    photonEnergyCorrector_->calculate(evt, newCandidate, subdet, *vertexHandle, es);
    if (candidateP4type_ == "fromEcalEnergy") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::ecal_photons));
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
    } else if (candidateP4type_ == "fromRegression1") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression1));
      newCandidate.setCandidateP4type(reco::Photon::regression1);
    } else if (candidateP4type_ == "fromRegression2") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
      newCandidate.setCandidateP4type(reco::Photon::regression2);
    } else if (candidateP4type_ == "fromRefinedSCRegression") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression2));
      newCandidate.setCandidateP4type(reco::Photon::regression2);
    }

    //    std::cout << " GEDPhotonProducer  pf based isolation  chargedHadron " << newCandidate.chargedHadronIso() << " neutralHadron " <<  newCandidate.neutralHadronIso() << " Photon " <<  newCandidate.photonIso() << std::endl;
    //std::cout << " GEDPhotonProducer from candidate HoE with towers in a cone " << newCandidate.hadronicOverEm()  << "  " <<  newCandidate.hadronicDepth1OverEm()  << " " <<  newCandidate.hadronicDepth2OverEm()  << std::endl;
    //std::cout << " GEDPhotonProducer from candidate  of HoE with towers behind the BCs " <<  newCandidate.hadTowOverEm()  << "  " << newCandidate.hadTowDepth1OverEm() << " " << newCandidate.hadTowDepth2OverEm() << std::endl;
    //std::cout << " standard p4 before " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;
    //std::cout << " type " <<newCandidate.getCandidateP4type() <<  " standard p4 after " << newCandidate.p4() << " energy " << newCandidate.energy() << std::endl;
    //std::cout << " final p4 " << newCandidate.p4() << " energy " << newCandidate.energy() <<  std::endl;

    outputPhotonCollection.push_back(newCandidate);
  }
}
