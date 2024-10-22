/** \class PhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMVABasedHaloTagger.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <vector>

// PhotonProducer inherits from EDProducer, so it can be a module:
class PhotonProducer : public edm::stream::EDProducer<> {
public:
  PhotonProducer(const edm::ParameterSet& ps);

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                            const CaloTopology* topology,
                            const HcalPFCuts* hcalCuts,
                            const EcalRecHitCollection* ecalBarrelHits,
                            const EcalRecHitCollection* ecalEndcapHits,
                            ElectronHcalHelper const& hcalHelperCone,
                            ElectronHcalHelper const& hcalHelperBc,
                            reco::VertexCollection& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC);

  // std::string PhotonCoreCollection_;
  std::string PhotonCollection_;
  edm::EDGetTokenT<reco::PhotonCoreCollection> photonCoreProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits_;
  edm::EDGetTokenT<reco::VertexCollection> vertexProducer_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> topologyToken_;

  //AA
  //Flags and severities to be excluded from calculations

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double hOverEConeSize_;
  double highEt_;
  double minR9Barrel_;
  double minR9Endcap_;
  bool runMIPTagger_;
  bool runMVABasedHaloTagger_;

  bool validConversions_;

  bool usePrimaryVertex_;

  PositionCalc posCalculator_;

  bool validPixelSeeds_;
  PhotonIsolationCalculator photonIsolationCalculator_;

  //MIP
  const PhotonMIPHaloTagger photonMIPHaloTagger_;
  //MVA based Halo tagger for the EE photons
  std::unique_ptr<const PhotonMVABasedHaloTagger> photonMVABasedHaloTagger_ = nullptr;

  std::vector<double> preselCutValuesBarrel_;
  std::vector<double> preselCutValuesEndcap_;

  PhotonEnergyCorrector photonEnergyCorrector_;
  std::string candidateP4type_;

  // additional configuration and helpers
  std::unique_ptr<ElectronHcalHelper> hcalHelperCone_;
  std::unique_ptr<ElectronHcalHelper> hcalHelperBc_;
  bool hcalRun2EffDepth_;

  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  bool cutsFromDB_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonProducer);

PhotonProducer::PhotonProducer(const edm::ParameterSet& config)
    : caloGeomToken_(esConsumes()),
      topologyToken_(esConsumes()),
      flagsexclEB_{StringToEnumValue<EcalRecHit::Flags>(
          config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB"))},
      flagsexclEE_{StringToEnumValue<EcalRecHit::Flags>(
          config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE"))},
      severitiesexclEB_{StringToEnumValue<EcalSeverityLevel::SeverityLevel>(
          config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEB"))},
      severitiesexclEE_{StringToEnumValue<EcalSeverityLevel::SeverityLevel>(
          config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEE"))},
      photonIsolationCalculator_(config.getParameter<edm::ParameterSet>("isolationSumsCalculatorSet"),
                                 flagsexclEB_,
                                 flagsexclEE_,
                                 severitiesexclEB_,
                                 severitiesexclEE_,
                                 consumesCollector()),
      photonMIPHaloTagger_(config.getParameter<edm::ParameterSet>("mipVariableSet"), consumesCollector()),
      photonEnergyCorrector_(config, consumesCollector()) {
  // use configuration file to setup input/output collection names

  photonCoreProducer_ = consumes<reco::PhotonCoreCollection>(config.getParameter<edm::InputTag>("photonCoreProducer"));
  barrelEcalHits_ = consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_ = consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalHits"));
  vertexProducer_ = consumes<reco::VertexCollection>(config.getParameter<edm::InputTag>("primaryVertexProducer"));
  hbheRecHits_ = consumes<HBHERecHitCollection>(config.getParameter<edm::InputTag>("hbheRecHits"));
  hOverEConeSize_ = config.getParameter<double>("hOverEConeSize");
  highEt_ = config.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_ = config.getParameter<double>("minR9Barrel");
  minR9Endcap_ = config.getParameter<double>("minR9Endcap");
  usePrimaryVertex_ = config.getParameter<bool>("usePrimaryVertex");
  runMIPTagger_ = config.getParameter<bool>("runMIPTagger");
  runMVABasedHaloTagger_ = config.getParameter<bool>("runMVABasedHaloTagger");

  candidateP4type_ = config.getParameter<std::string>("candidateP4type");

  edm::ParameterSet posCalcParameters = config.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);

  //Retrieve HCAL PF thresholds - from config or from DB
  cutsFromDB_ = config.getParameter<bool>("usePFThresholdsFromDB");
  if (cutsFromDB_) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd>(edm::ESInputTag("", "withTopo"));
  }

  ElectronHcalHelper::Configuration cfgCone, cfgBc;
  cfgCone.hOverEConeSize = hOverEConeSize_;
  if (cfgCone.hOverEConeSize > 0) {
    cfgCone.onlyBehindCluster = false;
    cfgCone.checkHcalStatus = false;

    cfgCone.hbheRecHits = hbheRecHits_;

    cfgCone.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
    cfgCone.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
    cfgCone.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
    cfgCone.maxSeverityHE = cfgCone.maxSeverityHB;
  }

  cfgBc.hOverEConeSize = 0.;
  cfgBc.onlyBehindCluster = true;
  cfgBc.checkHcalStatus = false;

  cfgBc.hbheRecHits = hbheRecHits_;

  cfgBc.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
  cfgBc.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
  cfgBc.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
  cfgBc.maxSeverityHE = cfgBc.maxSeverityHB;

  hcalHelperCone_ = std::make_unique<ElectronHcalHelper>(cfgCone, consumesCollector());
  hcalHelperBc_ = std::make_unique<ElectronHcalHelper>(cfgBc, consumesCollector());

  hcalRun2EffDepth_ = config.getParameter<bool>("hcalRun2EffDepth");

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
  preselCutValuesBarrel_.push_back(config.getParameter<double>("minSCEtBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("maxHoverEBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("ecalRecHitSumEtOffsetBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("ecalRecHitSumEtSlopeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("hcalRecHitSumEtOffsetBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("hcalRecHitSumEtSlopeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("nTrackSolidConeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("nTrackHollowConeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("trackPtSumSolidConeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("trackPtSumHollowConeBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("sigmaIetaIetaCutBarrel"));
  //
  preselCutValuesEndcap_.push_back(config.getParameter<double>("minSCEtEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("maxHoverEEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("ecalRecHitSumEtOffsetEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("ecalRecHitSumEtSlopeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("hcalRecHitSumEtOffsetEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("hcalRecHitSumEtSlopeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("nTrackSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("nTrackHollowConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("trackPtSumSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("trackPtSumHollowConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("sigmaIetaIetaCutEndcap"));
  //

  // Register the product
  produces<reco::PhotonCollection>(PhotonCollection_);
}

void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  HcalPFCuts const* hcalCuts = nullptr;
  if (cutsFromDB_) {
    hcalCuts = &theEventSetup.getData(hcalCutsToken_);
  }
  using namespace edm;
  //  nEvt_++;

  reco::PhotonCollection outputPhotonCollection;
  auto outputPhotonCollection_p = std::make_unique<reco::PhotonCollection>();

  // Get the PhotonCore collection
  bool validPhotonCoreHandle = true;
  Handle<reco::PhotonCoreCollection> photonCoreHandle;
  theEvent.getByToken(photonCoreProducer_, photonCoreHandle);
  if (!photonCoreHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the photonCoreProducer";
    validPhotonCoreHandle = false;
  }

  // Get EcalRecHits
  bool validEcalRecHits = true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  EcalRecHitCollection barrelRecHits;
  theEvent.getByToken(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the barrelEcalHits";
    validEcalRecHits = false;
  }
  if (validEcalRecHits)
    barrelRecHits = *(barrelHitHandle.product());

  Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByToken(endcapEcalHits_, endcapHitHandle);
  EcalRecHitCollection endcapRecHits;
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the endcapEcalHits";
    validEcalRecHits = false;
  }
  if (validEcalRecHits)
    endcapRecHits = *(endcapHitHandle.product());

  const CaloTopology* topology = &theEventSetup.getData(topologyToken_);

  // prepare access to hcal data
  hcalHelperCone_->beginEvent(theEvent, theEventSetup);
  hcalHelperBc_->beginEvent(theEvent, theEventSetup);

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  bool validVertex = true;
  if (usePrimaryVertex_) {
    theEvent.getByToken(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      edm::LogWarning("PhotonProducer") << "Error! Can't get the product primary Vertex Collection "
                                        << "\n";
      validVertex = false;
    }
    if (validVertex)
      vertexCollection = *(vertexHandle.product());
  }

  int iSC = 0;  // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  if (validPhotonCoreHandle)
    fillPhotonCollection(theEvent,
                         theEventSetup,
                         photonCoreHandle,
                         topology,
                         hcalCuts,
                         &barrelRecHits,
                         &endcapRecHits,
                         *hcalHelperCone_,
                         *hcalHelperBc_,
                         vertexCollection,
                         outputPhotonCollection,
                         iSC);

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(), outputPhotonCollection.end());

  // go back to run2-like 2 effective depths if desired - depth 1 is the normal depth 1, depth 2 is the sum over the rest
  if (hcalRun2EffDepth_) {
    for (auto& pho : *outputPhotonCollection_p)
      pho.hcalToRun2EffDepth();
  }

  theEvent.put(std::move(outputPhotonCollection_p), PhotonCollection_);
}

void PhotonProducer::fillPhotonCollection(edm::Event& evt,
                                          edm::EventSetup const& es,
                                          const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                                          const CaloTopology* topology,
                                          const HcalPFCuts* hcalCuts,
                                          const EcalRecHitCollection* ecalBarrelHits,
                                          const EcalRecHitCollection* ecalEndcapHits,
                                          ElectronHcalHelper const& hcalHelperCone,
                                          ElectronHcalHelper const& hcalHelperBc,
                                          reco::VertexCollection& vertexCollection,
                                          reco::PhotonCollection& outputPhotonCollection,
                                          int& iSC) {
  // get the geometry from the event setup:
  const CaloGeometry* geometry = &es.getData(caloGeomToken_);
  const CaloSubdetectorGeometry* subDetGeometry = nullptr;
  const CaloSubdetectorGeometry* geometryES = geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const EcalRecHitCollection* hits = nullptr;
  std::vector<double> preselCutValues;
  float minR9 = 0;

  photonEnergyCorrector_.init(es);

  std::vector<int> flags_, severitiesexcl_;

  for (unsigned int lSC = 0; lSC < photonCoreHandle->size(); lSC++) {
    reco::PhotonCoreRef coreRef(reco::PhotonCoreRef(photonCoreHandle, lSC));
    reco::SuperClusterRef scRef = coreRef->superCluster();
    iSC++;

    int subdet = scRef->seed()->hitsAndFractions()[0].first.subdetId();
    subDetGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, subdet);

    if (subdet == EcalBarrel) {
      preselCutValues = preselCutValuesBarrel_;
      minR9 = minR9Barrel_;
      hits = ecalBarrelHits;
      flags_ = flagsexclEB_;
      severitiesexcl_ = severitiesexclEB_;
    } else if (subdet == EcalEndcap) {
      preselCutValues = preselCutValuesEndcap_;
      minR9 = minR9Endcap_;
      hits = ecalEndcapHits;
      flags_ = flagsexclEE_;
      severitiesexcl_ = severitiesexclEE_;
    } else {
      edm::LogWarning("") << "PhotonProducer: do not know if it is a barrel or endcap SuperCluster";
    }
    if (hits == nullptr)
      continue;

    // SC energy preselection
    if (scRef->energy() / cosh(scRef->eta()) <= preselCutValues[0])
      continue;

    // recalculate position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint unconvPos =
        posCalculator_.Calculate_Location(scRef->seed()->hitsAndFractions(), hits, subDetGeometry, geometryES);

    float maxXtal = EcalClusterTools::eMax(*(scRef->seed()), &(*hits));
    //AA
    //Change these to consider severity level of hits
    float e1x5 = EcalClusterTools::e1x5(*(scRef->seed()), &(*hits), &(*topology));
    float e2x5 = EcalClusterTools::e2x5Max(*(scRef->seed()), &(*hits), &(*topology));
    float e3x3 = EcalClusterTools::e3x3(*(scRef->seed()), &(*hits), &(*topology));
    float e5x5 = EcalClusterTools::e5x5(*(scRef->seed()), &(*hits), &(*topology));
    const auto& cov = EcalClusterTools::covariances(*(scRef->seed()), &(*hits), &(*topology), geometry);
    const auto& locCov = EcalClusterTools::localCovariances(*(scRef->seed()), &(*hits), &(*topology));

    float sigmaEtaEta = sqrt(cov[0]);
    float sigmaIetaIeta = sqrt(locCov[0]);
    float r9 = e3x3 / (scRef->rawEnergy());

    float full5x5_maxXtal = noZS::EcalClusterTools::eMax(*(scRef->seed()), &(*hits));
    //AA
    //Change these to consider severity level of hits
    float full5x5_e1x5 = noZS::EcalClusterTools::e1x5(*(scRef->seed()), &(*hits), &(*topology));
    float full5x5_e2x5 = noZS::EcalClusterTools::e2x5Max(*(scRef->seed()), &(*hits), &(*topology));
    float full5x5_e3x3 = noZS::EcalClusterTools::e3x3(*(scRef->seed()), &(*hits), &(*topology));
    float full5x5_e5x5 = noZS::EcalClusterTools::e5x5(*(scRef->seed()), &(*hits), &(*topology));
    const auto& full5x5_cov = noZS::EcalClusterTools::covariances(*(scRef->seed()), &(*hits), &(*topology), geometry);
    const auto& full5x5_locCov = noZS::EcalClusterTools::localCovariances(*(scRef->seed()), &(*hits), &(*topology));

    float full5x5_sigmaEtaEta = sqrt(full5x5_cov[0]);
    float full5x5_sigmaIetaIeta = sqrt(full5x5_locCov[0]);

    // compute position of ECAL shower
    math::XYZPoint caloPosition;
    if (r9 > minR9) {
      caloPosition = unconvPos;
    } else {
      caloPosition = scRef->position();
    }

    //// energy determination -- Default to create the candidate. Afterwards corrections are applied
    double photonEnergy = 1.;
    math::XYZPoint vtx(0., 0., 0.);
    if (!vertexCollection.empty())
      vtx = vertexCollection.begin()->position();
    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    math::XYZVector momentum = direction.unit();

    // Create dummy candidate with unit momentum and zero energy to allow setting of all variables. The energy is set for last.
    math::XYZTLorentzVectorD p4(momentum.x(), momentum.y(), momentum.z(), photonEnergy);
    reco::Photon newCandidate(p4, caloPosition, coreRef, vtx);

    // Calculate fiducial flags and isolation variable. Blocked are filled from the isolationCalculator
    reco::Photon::FiducialFlags fiducialFlags;
    reco::Photon::IsolationVariables isolVarR03, isolVarR04;
    photonIsolationCalculator_.calculate(&newCandidate, evt, es, fiducialFlags, isolVarR04, isolVarR03, hcalCuts);
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
      showerShape.hcalOverEcal[id] = hcalHelperCone.hcalESum(*scRef, id + 1, hcalCuts) / scRef->energy();
      showerShape.hcalOverEcalBc[id] = hcalHelperBc.hcalESum(*scRef, id + 1, hcalCuts) / scRef->energy();
    }
    showerShape.hcalTowersBehindClusters = hcalHelperBc.hcalTowersBehindClusters(*scRef);
    showerShape.pre7DepthHcal = false;
    newCandidate.setShowerShapeVariables(showerShape);

    /// fill full5x5 shower shape block
    reco::Photon::ShowerShape full5x5_showerShape;
    full5x5_showerShape.e1x5 = full5x5_e1x5;
    full5x5_showerShape.e2x5 = full5x5_e2x5;
    full5x5_showerShape.e3x3 = full5x5_e3x3;
    full5x5_showerShape.e5x5 = full5x5_e5x5;
    full5x5_showerShape.maxEnergyXtal = full5x5_maxXtal;
    full5x5_showerShape.sigmaEtaEta = full5x5_sigmaEtaEta;
    full5x5_showerShape.sigmaIetaIeta = full5x5_sigmaIetaIeta;
    for (uint id = 0; id < full5x5_showerShape.hcalOverEcal.size(); ++id) {
      full5x5_showerShape.hcalOverEcal[id] = hcalHelperCone.hcalESum(*scRef, id + 1, hcalCuts) / full5x5_e5x5;
      full5x5_showerShape.hcalOverEcalBc[id] = hcalHelperBc.hcalESum(*scRef, id + 1, hcalCuts) / full5x5_e5x5;
    }
    full5x5_showerShape.hcalTowersBehindClusters = hcalHelperBc.hcalTowersBehindClusters(*scRef);
    full5x5_showerShape.pre7DepthHcal = false;
    newCandidate.full5x5_setShowerShapeVariables(full5x5_showerShape);

    /// get ecal photon specific corrected energy
    /// plus values from regressions     and store them in the Photon
    // Photon candidate takes by default (set in photons_cfi.py)  a 4-momentum derived from the ecal photon-specific corrections.
    photonEnergyCorrector_.calculate(evt, newCandidate, subdet, vertexCollection, es);
    if (candidateP4type_ == "fromEcalEnergy") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::ecal_photons));
      newCandidate.setCandidateP4type(reco::Photon::ecal_photons);
    } else if (candidateP4type_ == "fromRegression") {
      newCandidate.setP4(newCandidate.p4(reco::Photon::regression1));
      newCandidate.setCandidateP4type(reco::Photon::regression1);
    }

    // fill MIP Vairables for Halo: Block for MIP are filled from PhotonMIPHaloTagger
    if (subdet == EcalBarrel && runMIPTagger_) {
      auto mipVar = photonMIPHaloTagger_.mipCalculate(newCandidate, evt, es);
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
