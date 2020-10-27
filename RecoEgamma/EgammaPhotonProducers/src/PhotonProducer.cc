#include <vector>
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : photonEnergyCorrector_(config, consumesCollector()) {
  // use onfiguration file to setup input/output collection names

  photonCoreProducer_ = consumes<reco::PhotonCoreCollection>(config.getParameter<edm::InputTag>("photonCoreProducer"));
  barrelEcalHits_ = consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_ = consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalHits"));
  vertexProducer_ = consumes<reco::VertexCollection>(config.getParameter<edm::InputTag>("primaryVertexProducer"));
  hcalTowers_ = consumes<CaloTowerCollection>(config.getParameter<edm::InputTag>("hcalTowers"));
  hOverEConeSize_ = config.getParameter<double>("hOverEConeSize");
  highEt_ = config.getParameter<double>("highEt");
  // R9 value to decide converted/unconverted
  minR9Barrel_ = config.getParameter<double>("minR9Barrel");
  minR9Endcap_ = config.getParameter<double>("minR9Endcap");
  usePrimaryVertex_ = config.getParameter<bool>("usePrimaryVertex");
  runMIPTagger_ = config.getParameter<bool>("runMIPTagger");

  candidateP4type_ = config.getParameter<std::string>("candidateP4type");

  edm::ParameterSet posCalcParameters = config.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);

  //AA
  //Flags and Severities to be excluded from photon calculations
  const std::vector<std::string> flagnamesEB =
      config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB");

  const std::vector<std::string> flagnamesEE =
      config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const std::vector<std::string> severitynamesEB =
      config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const std::vector<std::string> severitynamesEE =
      config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

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
  preselCutValuesBarrel_.push_back(config.getParameter<double>("hcalTowerSumEtOffsetBarrel"));
  preselCutValuesBarrel_.push_back(config.getParameter<double>("hcalTowerSumEtSlopeBarrel"));
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
  preselCutValuesEndcap_.push_back(config.getParameter<double>("hcalTowerSumEtOffsetEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("hcalTowerSumEtSlopeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("nTrackSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("nTrackHollowConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("trackPtSumSolidConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("trackPtSumHollowConeEndcap"));
  preselCutValuesEndcap_.push_back(config.getParameter<double>("sigmaIetaIetaCutEndcap"));
  //

  edm::ParameterSet isolationSumsCalculatorSet = config.getParameter<edm::ParameterSet>("isolationSumsCalculatorSet");
  photonIsolationCalculator_.setup(
      isolationSumsCalculatorSet, flagsexclEB_, flagsexclEE_, severitiesexclEB_, severitiesexclEE_, consumesCollector());

  edm::ParameterSet mipVariableSet = config.getParameter<edm::ParameterSet>("mipVariableSet");
  photonMIPHaloTagger_.setup(mipVariableSet, consumesCollector());

  // Register the product
  produces<reco::PhotonCollection>(PhotonCollection_);
}

void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
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

  //AA
  //Get the severity level object
  edm::ESHandle<EcalSeverityLevelAlgo> sevLv;
  theEventSetup.get<EcalSeverityLevelAlgoRcd>().get(sevLv);
  //

  // get Hcal towers collection
  auto const& hcalTowers = theEvent.get(hcalTowers_);

  edm::ESHandle<CaloTopology> pTopology;
  theEventSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology* topology = pTopology.product();

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
                         &barrelRecHits,
                         &endcapRecHits,
                         hcalTowers,
                         vertexCollection,
                         outputPhotonCollection,
                         iSC,
                         sevLv.product());

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(), outputPhotonCollection.end());
  theEvent.put(std::move(outputPhotonCollection_p), PhotonCollection_);
}

void PhotonProducer::fillPhotonCollection(edm::Event& evt,
                                          edm::EventSetup const& es,
                                          const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                                          const CaloTopology* topology,
                                          const EcalRecHitCollection* ecalBarrelHits,
                                          const EcalRecHitCollection* ecalEndcapHits,
                                          CaloTowerCollection const& hcalTowers,
                                          reco::VertexCollection& vertexCollection,
                                          reco::PhotonCollection& outputPhotonCollection,
                                          int& iSC,
                                          const EcalSeverityLevelAlgo* sevLv) {
  // get the geometry from the event setup:
  edm::ESHandle<CaloGeometry> caloGeomHandle;
  es.get<CaloGeometryRecord>().get(caloGeomHandle);

  const CaloGeometry* geometry = caloGeomHandle.product();
  const CaloSubdetectorGeometry* subDetGeometry = nullptr;
  const CaloSubdetectorGeometry* geometryES = caloGeomHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
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
    subDetGeometry = caloGeomHandle->getSubdetectorGeometry(DetId::Ecal, subdet);

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
    // calculate HoE

    EgammaTowerIsolation towerIso1(hOverEConeSize_, 0., 0., 1, &hcalTowers);
    EgammaTowerIsolation towerIso2(hOverEConeSize_, 0., 0., 2, &hcalTowers);
    double HoE1 = towerIso1.getTowerESum(&(*scRef)) / scRef->energy();
    double HoE2 = towerIso2.getTowerESum(&(*scRef)) / scRef->energy();

    edm::ESHandle<CaloTowerConstituentsMap> ctmaph;
    es.get<CaloGeometryRecord>().get(ctmaph);

    auto towersBehindCluster = egamma::towersOf(*scRef, *ctmaph);
    float hcalDepth1OverEcalBc = egamma::depth1HcalESum(towersBehindCluster, hcalTowers) / scRef->energy();
    float hcalDepth2OverEcalBc = egamma::depth2HcalESum(towersBehindCluster, hcalTowers) / scRef->energy();

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
    std::vector<float> cov = EcalClusterTools::covariances(*(scRef->seed()), &(*hits), &(*topology), geometry);
    std::vector<float> locCov = EcalClusterTools::localCovariances(*(scRef->seed()), &(*hits), &(*topology));

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
    std::vector<float> full5x5_cov =
        noZS::EcalClusterTools::covariances(*(scRef->seed()), &(*hits), &(*topology), geometry);
    std::vector<float> full5x5_locCov =
        noZS::EcalClusterTools::localCovariances(*(scRef->seed()), &(*hits), &(*topology));

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
    photonIsolationCalculator_.calculate(&newCandidate, evt, es, fiducialFlags, isolVarR04, isolVarR03);
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
    showerShape.hcalTowersBehindClusters = towersBehindCluster;
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
    reco::Photon::MIPVariables mipVar;
    if (subdet == EcalBarrel && runMIPTagger_) {
      photonMIPHaloTagger_.MIPcalculate(&newCandidate, evt, es, mipVar);
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
