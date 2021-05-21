/** \class PhotonIsolationCalculator
 *  Determine and Set quality information on Photon Objects
 *
 *  \author A. Askew, N. Marinelli, M.B. Anderson
 */

#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

void PhotonIsolationCalculator::setup(const edm::ParameterSet& conf,
                                      std::vector<int> const& flagsEB,
                                      std::vector<int> const& flagsEE,
                                      std::vector<int> const& severitiesEB,
                                      std::vector<int> const& severitiesEE,
                                      edm::ConsumesCollector&& iC) {
  trackInputTag_ = iC.consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("trackProducer"));
  beamSpotProducerTag_ = iC.consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotProducer"));
  barrelecalCollection_ =
      iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  endcapecalCollection_ =
      iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));
  auto hcRHC = conf.getParameter<edm::InputTag>("HcalRecHitCollection");
  if (not hcRHC.label().empty())
    hcalCollection_ = iC.consumes<CaloTowerCollection>(hcRHC);

  //  gsfRecoInputTag_ = conf.getParameter<edm::InputTag>("GsfRecoCollection");
  modulePhiBoundary_ = conf.getParameter<double>("modulePhiBoundary");
  moduleEtaBoundary_ = conf.getParameter<std::vector<double> >("moduleEtaBoundary");
  //
  vetoClusteredEcalHits_ = conf.getParameter<bool>("vetoClustered");
  useNumCrystals_ = conf.getParameter<bool>("useNumCrystals");

  /// Isolation parameters for barrel and for two different cone sizes
  int i = 0;
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("isolationtrackThresholdA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("longImpactParameterA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("transImpactParameterA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceA_Barrel"));

  i = 0;
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtA_Barrel"));

  i = 0;
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalTowerInnerRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalTowerOuterRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalTowerThreshEA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerInnerRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerOuterRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerThreshEA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerInnerRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerOuterRadiusA_Barrel"));
  hcalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerThreshEA_Barrel"));

  i = 0;
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("isolationtrackThresholdB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("longImpactParameterB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("transImpactParameterB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceB_Barrel"));

  i = 0;
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtB_Barrel"));

  i = 0;
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalTowerInnerRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalTowerOuterRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalTowerThreshEB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerInnerRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerOuterRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerThreshEB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerInnerRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerOuterRadiusB_Barrel"));
  hcalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerThreshEB_Barrel"));

  /// Isolation parameters for Endcap and for two different cone sizes
  i = 0;
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("isolationtrackThresholdA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("longImpactParameterA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("transImpactParameterA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceA_Endcap"));

  i = 0;
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtA_Endcap"));

  i = 0;
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalTowerInnerRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalTowerOuterRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalTowerThreshEA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerInnerRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerOuterRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth1TowerThreshEA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerInnerRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerOuterRadiusA_Endcap"));
  hcalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("HcalDepth2TowerThreshEA_Endcap"));

  i = 0;
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("isolationtrackThresholdB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("longImpactParameterB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("transImpactParameterB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceB_Endcap"));

  i = 0;
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtB_Endcap"));

  i = 0;
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalTowerInnerRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalTowerOuterRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalTowerThreshEB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerInnerRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerOuterRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth1TowerThreshEB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerInnerRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerOuterRadiusB_Endcap"));
  hcalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("HcalDepth2TowerThreshEB_Endcap"));

  //Pick up the variables for the spike removal
  flagsEB_ = flagsEB;
  flagsEE_ = flagsEE;
  severityExclEB_ = severitiesEB;
  severityExclEE_ = severitiesEE;
}

void PhotonIsolationCalculator::calculate(const reco::Photon* pho,
                                          const edm::Event& e,
                                          const edm::EventSetup& es,
                                          reco::Photon::FiducialFlags& phofid,
                                          reco::Photon::IsolationVariables& phoisolR1,
                                          reco::Photon::IsolationVariables& phoisolR2) const {
  //Get fiducial flags. This does not really belong here
  bool isEBPho = false;
  bool isEEPho = false;
  bool isEBEtaGap = false;
  bool isEBPhiGap = false;
  bool isEERingGap = false;
  bool isEEDeeGap = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBEtaGap, isEBPhiGap, isEERingGap, isEEDeeGap, isEBEEGap);
  phofid.isEB = isEBPho;
  phofid.isEE = isEEPho;
  phofid.isEBEtaGap = isEBEtaGap;
  phofid.isEBPhiGap = isEBPhiGap;
  phofid.isEERingGap = isEERingGap;
  phofid.isEEDeeGap = isEEDeeGap;
  phofid.isEBEEGap = isEBEEGap;

  // Calculate isolation variables. cone sizes and thresholds
  // are set for Barrel and endcap separately

  reco::SuperClusterRef scRef = pho->superCluster();
  const reco::BasicCluster& seedCluster = *(scRef->seed());
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first;
  int detector = seedXtalId.subdetId();

  //Isolation parameters variables
  double photonEcalRecHitConeInnerRadiusA_;
  double photonEcalRecHitConeOuterRadiusA_;
  double photonEcalRecHitEtaSliceA_;
  double photonEcalRecHitThreshEA_;
  double photonEcalRecHitThreshEtA_;
  double photonHcalTowerConeInnerRadiusA_;
  double photonHcalTowerConeOuterRadiusA_;
  double photonHcalTowerThreshEA_;
  double photonHcalDepth1TowerConeInnerRadiusA_;
  double photonHcalDepth1TowerConeOuterRadiusA_;
  double photonHcalDepth1TowerThreshEA_;
  double photonHcalDepth2TowerConeInnerRadiusA_;
  double photonHcalDepth2TowerConeOuterRadiusA_;
  double photonHcalDepth2TowerThreshEA_;
  double trackConeOuterRadiusA_;
  double trackConeInnerRadiusA_;
  double isolationtrackThresholdA_;
  double isolationtrackEtaSliceA_;
  double trackLipRadiusA_;
  double trackD0RadiusA_;
  double photonEcalRecHitConeInnerRadiusB_;
  double photonEcalRecHitConeOuterRadiusB_;
  double photonEcalRecHitEtaSliceB_;
  double photonEcalRecHitThreshEB_;
  double photonEcalRecHitThreshEtB_;
  double photonHcalTowerConeInnerRadiusB_;
  double photonHcalTowerConeOuterRadiusB_;
  double photonHcalTowerThreshEB_;
  double photonHcalDepth1TowerConeInnerRadiusB_;
  double photonHcalDepth1TowerConeOuterRadiusB_;
  double photonHcalDepth1TowerThreshEB_;
  double photonHcalDepth2TowerConeInnerRadiusB_;
  double photonHcalDepth2TowerConeOuterRadiusB_;
  double photonHcalDepth2TowerThreshEB_;
  double trackConeOuterRadiusB_;
  double trackConeInnerRadiusB_;
  double isolationtrackThresholdB_;
  double isolationtrackEtaSliceB_;
  double trackLipRadiusB_;
  double trackD0RadiusB_;

  if (detector == EcalBarrel) {
    trackConeOuterRadiusA_ = trkIsoBarrelRadiusA_[0];
    trackConeInnerRadiusA_ = trkIsoBarrelRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoBarrelRadiusA_[2];
    trackLipRadiusA_ = trkIsoBarrelRadiusA_[3];
    trackD0RadiusA_ = trkIsoBarrelRadiusA_[4];
    isolationtrackEtaSliceA_ = trkIsoBarrelRadiusA_[5];

    photonEcalRecHitConeInnerRadiusA_ = ecalIsoBarrelRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_ = ecalIsoBarrelRadiusA_[1];
    photonEcalRecHitEtaSliceA_ = ecalIsoBarrelRadiusA_[2];
    photonEcalRecHitThreshEA_ = ecalIsoBarrelRadiusA_[3];
    photonEcalRecHitThreshEtA_ = ecalIsoBarrelRadiusA_[4];

    photonHcalTowerConeInnerRadiusA_ = hcalIsoBarrelRadiusA_[0];
    photonHcalTowerConeOuterRadiusA_ = hcalIsoBarrelRadiusA_[1];
    photonHcalTowerThreshEA_ = hcalIsoBarrelRadiusA_[2];
    photonHcalDepth1TowerConeInnerRadiusA_ = hcalIsoBarrelRadiusA_[3];
    photonHcalDepth1TowerConeOuterRadiusA_ = hcalIsoBarrelRadiusA_[4];
    photonHcalDepth1TowerThreshEA_ = hcalIsoBarrelRadiusA_[5];
    photonHcalDepth2TowerConeInnerRadiusA_ = hcalIsoBarrelRadiusA_[6];
    photonHcalDepth2TowerConeOuterRadiusA_ = hcalIsoBarrelRadiusA_[7];
    photonHcalDepth2TowerThreshEA_ = hcalIsoBarrelRadiusA_[8];

    trackConeOuterRadiusB_ = trkIsoBarrelRadiusB_[0];
    trackConeInnerRadiusB_ = trkIsoBarrelRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoBarrelRadiusB_[2];
    trackLipRadiusB_ = trkIsoBarrelRadiusB_[3];
    trackD0RadiusB_ = trkIsoBarrelRadiusB_[4];
    isolationtrackEtaSliceB_ = trkIsoBarrelRadiusB_[5];

    photonEcalRecHitConeInnerRadiusB_ = ecalIsoBarrelRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_ = ecalIsoBarrelRadiusB_[1];
    photonEcalRecHitEtaSliceB_ = ecalIsoBarrelRadiusB_[2];
    photonEcalRecHitThreshEB_ = ecalIsoBarrelRadiusB_[3];
    photonEcalRecHitThreshEtB_ = ecalIsoBarrelRadiusB_[4];

    photonHcalTowerConeInnerRadiusB_ = hcalIsoBarrelRadiusB_[0];
    photonHcalTowerConeOuterRadiusB_ = hcalIsoBarrelRadiusB_[1];
    photonHcalTowerThreshEB_ = hcalIsoBarrelRadiusB_[2];
    photonHcalDepth1TowerConeInnerRadiusB_ = hcalIsoBarrelRadiusB_[3];
    photonHcalDepth1TowerConeOuterRadiusB_ = hcalIsoBarrelRadiusB_[4];
    photonHcalDepth1TowerThreshEB_ = hcalIsoBarrelRadiusB_[5];
    photonHcalDepth2TowerConeInnerRadiusB_ = hcalIsoBarrelRadiusB_[6];
    photonHcalDepth2TowerConeOuterRadiusB_ = hcalIsoBarrelRadiusB_[7];
    photonHcalDepth2TowerThreshEB_ = hcalIsoBarrelRadiusB_[8];

  } else {
    // detector==EcalEndcap

    trackConeOuterRadiusA_ = trkIsoEndcapRadiusA_[0];
    trackConeInnerRadiusA_ = trkIsoEndcapRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoEndcapRadiusA_[2];
    trackLipRadiusA_ = trkIsoEndcapRadiusA_[3];
    trackD0RadiusA_ = trkIsoEndcapRadiusA_[4];
    isolationtrackEtaSliceA_ = trkIsoEndcapRadiusA_[5];

    photonEcalRecHitConeInnerRadiusA_ = ecalIsoEndcapRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_ = ecalIsoEndcapRadiusA_[1];
    photonEcalRecHitEtaSliceA_ = ecalIsoEndcapRadiusA_[2];
    photonEcalRecHitThreshEA_ = ecalIsoEndcapRadiusA_[3];
    photonEcalRecHitThreshEtA_ = ecalIsoEndcapRadiusA_[4];

    photonHcalTowerConeInnerRadiusA_ = hcalIsoEndcapRadiusA_[0];
    photonHcalTowerConeOuterRadiusA_ = hcalIsoEndcapRadiusA_[1];
    photonHcalTowerThreshEA_ = hcalIsoEndcapRadiusA_[2];
    photonHcalDepth1TowerConeInnerRadiusA_ = hcalIsoEndcapRadiusA_[3];
    photonHcalDepth1TowerConeOuterRadiusA_ = hcalIsoEndcapRadiusA_[4];
    photonHcalDepth1TowerThreshEA_ = hcalIsoEndcapRadiusA_[5];
    photonHcalDepth2TowerConeInnerRadiusA_ = hcalIsoEndcapRadiusA_[6];
    photonHcalDepth2TowerConeOuterRadiusA_ = hcalIsoEndcapRadiusA_[7];
    photonHcalDepth2TowerThreshEA_ = hcalIsoEndcapRadiusA_[8];

    trackConeOuterRadiusB_ = trkIsoEndcapRadiusB_[0];
    trackConeInnerRadiusB_ = trkIsoEndcapRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoEndcapRadiusB_[2];
    trackLipRadiusB_ = trkIsoEndcapRadiusB_[3];
    trackD0RadiusB_ = trkIsoEndcapRadiusB_[4];
    isolationtrackEtaSliceB_ = trkIsoEndcapRadiusB_[5];

    photonEcalRecHitConeInnerRadiusB_ = ecalIsoEndcapRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_ = ecalIsoEndcapRadiusB_[1];
    photonEcalRecHitEtaSliceB_ = ecalIsoEndcapRadiusB_[2];
    photonEcalRecHitThreshEB_ = ecalIsoEndcapRadiusB_[3];
    photonEcalRecHitThreshEtB_ = ecalIsoEndcapRadiusB_[4];

    photonHcalTowerConeInnerRadiusB_ = hcalIsoEndcapRadiusB_[0];
    photonHcalTowerConeOuterRadiusB_ = hcalIsoEndcapRadiusB_[1];
    photonHcalTowerThreshEB_ = hcalIsoEndcapRadiusB_[2];
    photonHcalDepth1TowerConeInnerRadiusB_ = hcalIsoEndcapRadiusB_[3];
    photonHcalDepth1TowerConeOuterRadiusB_ = hcalIsoEndcapRadiusB_[4];
    photonHcalDepth1TowerThreshEB_ = hcalIsoEndcapRadiusB_[5];
    photonHcalDepth2TowerConeInnerRadiusB_ = hcalIsoEndcapRadiusB_[6];
    photonHcalDepth2TowerConeOuterRadiusB_ = hcalIsoEndcapRadiusB_[7];
    photonHcalDepth2TowerThreshEB_ = hcalIsoEndcapRadiusB_[8];
  }

  //Calculate hollow cone track isolation, CONE A
  int ntrkA = 0;
  double trkisoA = 0;
  calculateTrackIso(pho,
                    e,
                    trkisoA,
                    ntrkA,
                    isolationtrackThresholdA_,
                    trackConeOuterRadiusA_,
                    trackConeInnerRadiusA_,
                    isolationtrackEtaSliceA_,
                    trackLipRadiusA_,
                    trackD0RadiusA_);

  //Calculate solid cone track isolation, CONE A
  int sntrkA = 0;
  double strkisoA = 0;
  calculateTrackIso(pho,
                    e,
                    strkisoA,
                    sntrkA,
                    isolationtrackThresholdA_,
                    trackConeOuterRadiusA_,
                    0.,
                    isolationtrackEtaSliceA_,
                    trackLipRadiusA_,
                    trackD0RadiusA_);

  phoisolR1.nTrkHollowCone = ntrkA;
  phoisolR1.trkSumPtHollowCone = trkisoA;
  phoisolR1.nTrkSolidCone = sntrkA;
  phoisolR1.trkSumPtSolidCone = strkisoA;

  //Calculate hollow cone track isolation, CONE B
  int ntrkB = 0;
  double trkisoB = 0;
  calculateTrackIso(pho,
                    e,
                    trkisoB,
                    ntrkB,
                    isolationtrackThresholdB_,
                    trackConeOuterRadiusB_,
                    trackConeInnerRadiusB_,
                    isolationtrackEtaSliceB_,
                    trackLipRadiusB_,
                    trackD0RadiusB_);

  //Calculate solid cone track isolation, CONE B
  int sntrkB = 0;
  double strkisoB = 0;
  calculateTrackIso(pho,
                    e,
                    strkisoB,
                    sntrkB,
                    isolationtrackThresholdB_,
                    trackConeOuterRadiusB_,
                    0.,
                    isolationtrackEtaSliceB_,
                    trackLipRadiusB_,
                    trackD0RadiusB_);

  phoisolR2.nTrkHollowCone = ntrkB;
  phoisolR2.trkSumPtHollowCone = trkisoB;
  phoisolR2.nTrkSolidCone = sntrkB;
  phoisolR2.trkSumPtSolidCone = strkisoB;

  //   std::cout << "Output from solid cone track isolation: ";
  //   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;

  double EcalRecHitIsoA = calculateEcalRecHitIso(pho,
                                                 e,
                                                 es,
                                                 photonEcalRecHitConeOuterRadiusA_,
                                                 photonEcalRecHitConeInnerRadiusA_,
                                                 photonEcalRecHitEtaSliceA_,
                                                 photonEcalRecHitThreshEA_,
                                                 photonEcalRecHitThreshEtA_,
                                                 vetoClusteredEcalHits_,
                                                 useNumCrystals_);
  phoisolR1.ecalRecHitSumEt = EcalRecHitIsoA;

  double EcalRecHitIsoB = calculateEcalRecHitIso(pho,
                                                 e,
                                                 es,
                                                 photonEcalRecHitConeOuterRadiusB_,
                                                 photonEcalRecHitConeInnerRadiusB_,
                                                 photonEcalRecHitEtaSliceB_,
                                                 photonEcalRecHitThreshEB_,
                                                 photonEcalRecHitThreshEtB_,
                                                 vetoClusteredEcalHits_,
                                                 useNumCrystals_);
  phoisolR2.ecalRecHitSumEt = EcalRecHitIsoB;

  double HcalTowerIsoA = calculateHcalTowerIso(
      pho, e, es, photonHcalTowerConeOuterRadiusA_, photonHcalTowerConeInnerRadiusA_, photonHcalTowerThreshEA_, -1);
  phoisolR1.hcalTowerSumEt = HcalTowerIsoA;

  double HcalTowerIsoB = calculateHcalTowerIso(
      pho, e, es, photonHcalTowerConeOuterRadiusB_, photonHcalTowerConeInnerRadiusB_, photonHcalTowerThreshEB_, -1);
  phoisolR2.hcalTowerSumEt = HcalTowerIsoB;

  //// Hcal depth1

  double HcalDepth1TowerIsoA = calculateHcalTowerIso(pho,
                                                     e,
                                                     es,
                                                     photonHcalDepth1TowerConeOuterRadiusA_,
                                                     photonHcalDepth1TowerConeInnerRadiusA_,
                                                     photonHcalDepth1TowerThreshEA_,
                                                     1);
  phoisolR1.hcalDepth1TowerSumEt = HcalDepth1TowerIsoA;

  double HcalDepth1TowerIsoB = calculateHcalTowerIso(pho,
                                                     e,
                                                     es,
                                                     photonHcalDepth1TowerConeOuterRadiusB_,
                                                     photonHcalDepth1TowerConeInnerRadiusB_,
                                                     photonHcalDepth1TowerThreshEB_,
                                                     1);
  phoisolR2.hcalDepth1TowerSumEt = HcalDepth1TowerIsoB;

  //// Hcal depth2

  double HcalDepth2TowerIsoA = calculateHcalTowerIso(pho,
                                                     e,
                                                     es,
                                                     photonHcalDepth2TowerConeOuterRadiusA_,
                                                     photonHcalDepth2TowerConeInnerRadiusA_,
                                                     photonHcalDepth2TowerThreshEA_,
                                                     2);
  phoisolR1.hcalDepth2TowerSumEt = HcalDepth2TowerIsoA;

  double HcalDepth2TowerIsoB = calculateHcalTowerIso(pho,
                                                     e,
                                                     es,
                                                     photonHcalDepth2TowerConeOuterRadiusB_,
                                                     photonHcalDepth2TowerConeInnerRadiusB_,
                                                     photonHcalDepth2TowerThreshEB_,
                                                     2);
  phoisolR2.hcalDepth2TowerSumEt = HcalDepth2TowerIsoB;

  // New Hcal isolation based on the new H/E definition (towers behind the BCs in the SC are used to evaluated H)
  double HcalTowerBcIsoA =
      calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusA_, photonHcalTowerThreshEA_, -1);
  phoisolR1.hcalTowerSumEtBc = HcalTowerBcIsoA;

  double HcalTowerBcIsoB =
      calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusB_, photonHcalTowerThreshEB_, -1);
  phoisolR2.hcalTowerSumEtBc = HcalTowerBcIsoB;

  //// Hcal depth1

  double HcalDepth1TowerBcIsoA =
      calculateHcalTowerIso(pho, e, es, photonHcalDepth1TowerConeOuterRadiusA_, photonHcalDepth1TowerThreshEA_, 1);
  phoisolR1.hcalDepth1TowerSumEtBc = HcalDepth1TowerBcIsoA;

  double HcalDepth1TowerBcIsoB =
      calculateHcalTowerIso(pho, e, es, photonHcalDepth1TowerConeOuterRadiusB_, photonHcalDepth1TowerThreshEB_, 1);
  phoisolR2.hcalDepth1TowerSumEtBc = HcalDepth1TowerBcIsoB;

  //// Hcal depth2

  double HcalDepth2TowerBcIsoA =
      calculateHcalTowerIso(pho, e, es, photonHcalDepth2TowerConeOuterRadiusA_, photonHcalDepth2TowerThreshEA_, 2);
  phoisolR1.hcalDepth2TowerSumEtBc = HcalDepth2TowerBcIsoA;

  double HcalDepth2TowerBcIsoB =
      calculateHcalTowerIso(pho, e, es, photonHcalDepth2TowerConeOuterRadiusB_, photonHcalDepth2TowerThreshEB_, 2);
  phoisolR2.hcalDepth2TowerSumEtBc = HcalDepth2TowerBcIsoB;
}

void PhotonIsolationCalculator::classify(const reco::Photon* photon,
                                         bool& isEBPho,
                                         bool& isEEPho,
                                         bool& isEBEtaGap,
                                         bool& isEBPhiGap,
                                         bool& isEERingGap,
                                         bool& isEEDeeGap,
                                         bool& isEBEEGap) {
  const reco::CaloCluster& seedCluster = *(photon->superCluster()->seed());
  // following line is temporary until the D. Evans or F. Ferri submit the new tag for ClusterAlgos
  // DEfinitive will be
  // DetId seedXtalId = scRef->seed()->seed();
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first;
  int detector = seedXtalId.subdetId();

  //Set fiducial flags for this photon.
  double eta = photon->superCluster()->position().eta();
  double feta = fabs(eta);

  if (detector == EcalBarrel) {
    isEBPho = true;
    if (EBDetId::isNextToEtaBoundary(EBDetId(seedXtalId))) {
      if (fabs(feta - 1.479) < .1)
        isEBEEGap = true;
      else
        isEBEtaGap = true;
    }

    if (EBDetId::isNextToPhiBoundary(EBDetId(seedXtalId)))
      isEBPhiGap = true;

  } else if (detector == EcalEndcap) {
    isEEPho = true;
    if (EEDetId::isNextToRingBoundary(EEDetId(seedXtalId))) {
      if (fabs(feta - 1.479) < .1)
        isEBEEGap = true;
      else
        isEERingGap = true;
    }

    if (EEDetId::isNextToDBoundary(EEDetId(seedXtalId)))
      isEEDeeGap = true;
  }
}

void PhotonIsolationCalculator::calculateTrackIso(const reco::Photon* photon,
                                                  const edm::Event& e,
                                                  double& trkCone,
                                                  int& ntrkCone,
                                                  double pTThresh,
                                                  double RCone,
                                                  double RinnerCone,
                                                  double etaSlice,
                                                  double lip,
                                                  double d0) const {
  ntrkCone = 0;
  trkCone = 0;
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  e.getByToken(trackInputTag_, tracks);
  if (!tracks.isValid()) {
    return;
  }
  const reco::TrackCollection* trackCollection = tracks.product();
  //Photon Eta and Phi.  Hope these are correct.
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByToken(beamSpotProducerTag_, recoBeamSpotHandle);
  vertexBeamSpot = *recoBeamSpotHandle;

  PhotonTkIsolation phoIso(RCone,
                           RinnerCone,
                           etaSlice,
                           pTThresh,
                           lip,
                           d0,
                           trackCollection,
                           math::XYZPoint(vertexBeamSpot.x0(), vertexBeamSpot.y0(), vertexBeamSpot.z0()));

  std::pair<int, double> res = phoIso.getIso(photon);
  ntrkCone = res.first;
  trkCone = res.second;
}

double PhotonIsolationCalculator::calculateEcalRecHitIso(const reco::Photon* photon,
                                                         const edm::Event& iEvent,
                                                         const edm::EventSetup& iSetup,
                                                         double RCone,
                                                         double RConeInner,
                                                         double etaSlice,
                                                         double eMin,
                                                         double etMin,
                                                         bool vetoClusteredHits,
                                                         bool useNumXtals) const {
  edm::Handle<EcalRecHitCollection> ecalhitsCollEB;
  edm::Handle<EcalRecHitCollection> ecalhitsCollEE;
  iEvent.getByToken(endcapecalCollection_, ecalhitsCollEE);

  iEvent.getByToken(barrelecalCollection_, ecalhitsCollEB);

  const EcalRecHitCollection* rechitsCollectionEE_ = ecalhitsCollEE.product();
  const EcalRecHitCollection* rechitsCollectionEB_ = ecalhitsCollEB.product();

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);

  EgammaRecHitIsolation phoIsoEB(
      RCone, RConeInner, etaSlice, etMin, eMin, geoHandle, *rechitsCollectionEB_, sevLevel, DetId::Ecal);

  phoIsoEB.setVetoClustered(vetoClusteredHits);
  phoIsoEB.setUseNumCrystals(useNumXtals);
  phoIsoEB.doSeverityChecks(ecalhitsCollEB.product(), severityExclEB_);
  phoIsoEB.doFlagChecks(flagsEB_);
  double ecalIsolEB = phoIsoEB.getEtSum(photon);

  EgammaRecHitIsolation phoIsoEE(
      RCone, RConeInner, etaSlice, etMin, eMin, geoHandle, *rechitsCollectionEE_, sevLevel, DetId::Ecal);

  phoIsoEE.setVetoClustered(vetoClusteredHits);
  phoIsoEE.setUseNumCrystals(useNumXtals);
  phoIsoEE.doSeverityChecks(ecalhitsCollEE.product(), severityExclEE_);
  phoIsoEE.doFlagChecks(flagsEE_);

  double ecalIsolEE = phoIsoEE.getEtSum(photon);
  //  delete phoIso;
  double ecalIsol = ecalIsolEB + ecalIsolEE;

  return ecalIsol;
}

double PhotonIsolationCalculator::calculateHcalTowerIso(const reco::Photon* photon,
                                                        const edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup,
                                                        double RCone,
                                                        double RConeInner,
                                                        double eMin,
                                                        signed int depth) const {
  double hcalIsol = 0.;

  if (not hcalCollection_.isUninitialized()) {
    edm::Handle<CaloTowerCollection> hcalhitsCollH;
    iEvent.getByToken(hcalCollection_, hcalhitsCollH);

    const CaloTowerCollection* toww = hcalhitsCollH.product();

    //std::cout << "before iso call" << std::endl;
    EgammaTowerIsolation phoIso(RCone, RConeInner, eMin, depth, toww);
    hcalIsol = phoIso.getTowerEtSum(photon);
    //  delete phoIso;
    //std::cout << "after call" << std::endl;
  }

  return hcalIsol;
}

double PhotonIsolationCalculator::calculateHcalTowerIso(const reco::Photon* photon,
                                                        const edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup,
                                                        double RCone,
                                                        double eMin,
                                                        signed int depth) const {
  double hcalIsol = 0.;

  if (not hcalCollection_.isUninitialized()) {
    edm::Handle<CaloTowerCollection> hcalhitsCollH;
    iEvent.getByToken(hcalCollection_, hcalhitsCollH);

    const CaloTowerCollection* toww = hcalhitsCollH.product();

    //std::cout << "before iso call" << std::endl;
    EgammaTowerIsolation phoIso(RCone, 0., eMin, depth, toww);
    hcalIsol = phoIso.getTowerEtSum(photon, &(photon->hcalTowersBehindClusters()));
    //  delete phoIso;
    //std::cout << "after call" << std::endl;
  }

  return hcalIsol;
}
