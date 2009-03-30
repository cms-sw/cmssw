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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <string>
#include <TMath.h>


void PhotonIsolationCalculator::setup(const edm::ParameterSet& conf) {


  trackInputTag_ = conf.getParameter<edm::InputTag>("trackProducer");
  beamSpotProducerTag_ = conf.getParameter<edm::InputTag>("beamSpotProducer");
  barrelecalCollection_ = conf.getParameter<std::string>("barrelEcalRecHitCollection");
  barrelecalProducer_ = conf.getParameter<std::string>("barrelEcalRecHitProducer");
  endcapecalCollection_ = conf.getParameter<std::string>("endcapEcalRecHitCollection");
  endcapecalProducer_ = conf.getParameter<std::string>("endcapEcalRecHitProducer");
  hcalCollection_ = conf.getParameter<std::string>("HcalRecHitCollection");
  hcalProducer_ = conf.getParameter<std::string>("HcalRecHitProducer");

  //  gsfRecoInputTag_ = conf.getParameter<edm::InputTag>("GsfRecoCollection");
  modulePhiBoundary_ = conf.getParameter<double>("modulePhiBoundary");
  moduleEtaBoundary_ = conf.getParameter<std::vector<double> >("moduleEtaBoundary");

  /// Isolation parameters for barrel and for two different cone sizes
  trkIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("TrackConeOuterRadiusA_Barrel") );
  trkIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("TrackConeInnerRadiusA_Barrel") ) ;
  trkIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("isolationtrackThresholdA_Barrel") );
  trkIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("longImpactParameterA_Barrel") );
  trkIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("transImpactParameterA_Barrel") );


  ecalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("EcalRecHitInnerRadiusA_Barrel") );
  ecalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("EcalRecHitOuterRadiusA_Barrel") );
  ecalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("EcalRecHitEtaSliceA_Barrel") );
  ecalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("EcalRecHitThreshEA_Barrel") );
  ecalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("EcalRecHitThreshEtA_Barrel") );

  hcalIsoBarrelRadiusA_.push_back(  conf.getParameter<double>("HcalTowerInnerRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalTowerOuterRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalTowerThreshEA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerInnerRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerOuterRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerThreshEA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerInnerRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerOuterRadiusA_Barrel") );
  hcalIsoBarrelRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerThreshEA_Barrel") );


  trkIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("TrackConeOuterRadiusB_Barrel") );
  trkIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("TrackConeInnerRadiusB_Barrel") ) ;
  trkIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("isolationtrackThresholdB_Barrel") );
  trkIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("longImpactParameterB_Barrel") );
  trkIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("transImpactParameterB_Barrel") );


  ecalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("EcalRecHitInnerRadiusB_Barrel") );
  ecalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("EcalRecHitOuterRadiusB_Barrel") );
  ecalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("EcalRecHitEtaSliceB_Barrel") );
  ecalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("EcalRecHitThreshEB_Barrel") );
  ecalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("EcalRecHitThreshEtB_Barrel") );

  hcalIsoBarrelRadiusB_.push_back(  conf.getParameter<double>("HcalTowerInnerRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalTowerOuterRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalTowerThreshEB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerInnerRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerOuterRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerThreshEB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerInnerRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerOuterRadiusB_Barrel") );
  hcalIsoBarrelRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerThreshEB_Barrel") );

  /// Isolation parameters for Endcap and for two different cone sizes
  trkIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("TrackConeOuterRadiusA_Endcap") );
  trkIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("TrackConeInnerRadiusA_Endcap") ) ;
  trkIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("isolationtrackThresholdA_Endcap") );
  trkIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("longImpactParameterA_Endcap") );
  trkIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("transImpactParameterA_Endcap") );


  ecalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("EcalRecHitInnerRadiusA_Endcap") );
  ecalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("EcalRecHitOuterRadiusA_Endcap") );
  ecalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("EcalRecHitEtaSliceA_Endcap") );
  ecalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("EcalRecHitThreshEA_Endcap") );
  ecalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("EcalRecHitThreshEtA_Endcap") );

  hcalIsoEndcapRadiusA_.push_back(  conf.getParameter<double>("HcalTowerInnerRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalTowerOuterRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalTowerThreshEA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerInnerRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerOuterRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth1TowerThreshEA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerInnerRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerOuterRadiusA_Endcap") );
  hcalIsoEndcapRadiusA_.push_back( conf.getParameter<double>("HcalDepth2TowerThreshEA_Endcap") );


  trkIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("TrackConeOuterRadiusB_Endcap") );
  trkIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("TrackConeInnerRadiusB_Endcap") ) ;
  trkIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("isolationtrackThresholdB_Endcap") );
  trkIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("longImpactParameterB_Endcap") );
  trkIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("transImpactParameterB_Endcap") );


  ecalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("EcalRecHitInnerRadiusB_Endcap") );
  ecalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("EcalRecHitOuterRadiusB_Endcap") );
  ecalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("EcalRecHitEtaSliceB_Endcap") );
  ecalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("EcalRecHitThreshEB_Endcap") );
  ecalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("EcalRecHitThreshEtB_Endcap") );

  hcalIsoEndcapRadiusB_.push_back(  conf.getParameter<double>("HcalTowerInnerRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalTowerOuterRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalTowerThreshEB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerInnerRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerOuterRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth1TowerThreshEB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerInnerRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerOuterRadiusB_Endcap") );
  hcalIsoEndcapRadiusB_.push_back( conf.getParameter<double>("HcalDepth2TowerThreshEB_Endcap") );

}


void PhotonIsolationCalculator::calculate(const reco::Photon* pho,
				     const edm::Event& e,
				     const edm::EventSetup& es,
				     reco::Photon::FiducialFlags& phofid, 
				     reco::Photon::IsolationVariables& phoisolR1, 
				     reco::Photon::IsolationVariables& phoisolR2){


  //Get fiducial flags. This does not really belong here
  bool isEBPho   = false;
  bool isEEPho   = false;
  bool isEBGap   = false;
  bool isEEGap   = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap);
  phofid.isEB = isEBPho;
  phofid.isEE = isEEPho;
  phofid.isEBGap = isEBGap;
  phofid.isEEGap = isEEGap;
  phofid.isEBEEGap = isEBEEGap;
  
  // Calculate isolation variables. cone sizes and thresholds
  // are set for Barrel and endcap separately 

  reco::SuperClusterRef scRef=pho->superCluster();
  const reco::BasicCluster & seedCluster = *(scRef->seed()) ;
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;
  int detector = seedXtalId.subdetId() ;
  if (detector==EcalBarrel) {

    trackConeOuterRadiusA_    = trkIsoBarrelRadiusA_[0];
    trackConeInnerRadiusA_    = trkIsoBarrelRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoBarrelRadiusA_[2];
    trackLipRadiusA_          = trkIsoBarrelRadiusA_[3];
    trackD0RadiusA_           = trkIsoBarrelRadiusA_[4];

    photonEcalRecHitConeInnerRadiusA_  = ecalIsoBarrelRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_  = ecalIsoBarrelRadiusA_[1];
    photonEcalRecHitEtaSliceA_         = ecalIsoBarrelRadiusA_[2];
    photonEcalRecHitThreshEA_          = ecalIsoBarrelRadiusA_[3];
    photonEcalRecHitThreshEtA_         = ecalIsoBarrelRadiusA_[4];

    photonHcalTowerConeInnerRadiusA_       = hcalIsoBarrelRadiusA_[0];
    photonHcalTowerConeOuterRadiusA_       = hcalIsoBarrelRadiusA_[1];
    photonHcalTowerThreshEA_               = hcalIsoBarrelRadiusA_[2];
    photonHcalDepth1TowerConeInnerRadiusA_ = hcalIsoBarrelRadiusA_[3];
    photonHcalDepth1TowerConeOuterRadiusA_ = hcalIsoBarrelRadiusA_[4];
    photonHcalDepth1TowerThreshEA_         = hcalIsoBarrelRadiusA_[5];
    photonHcalDepth2TowerConeInnerRadiusA_ = hcalIsoBarrelRadiusA_[6];
    photonHcalDepth2TowerConeOuterRadiusA_ = hcalIsoBarrelRadiusA_[7];
    photonHcalDepth2TowerThreshEA_         = hcalIsoBarrelRadiusA_[8];


    trackConeOuterRadiusB_    = trkIsoBarrelRadiusB_[0];
    trackConeInnerRadiusB_    = trkIsoBarrelRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoBarrelRadiusB_[2];
    trackLipRadiusB_          = trkIsoBarrelRadiusB_[3];
    trackD0RadiusB_           = trkIsoBarrelRadiusB_[4];


    photonEcalRecHitConeInnerRadiusB_  = ecalIsoBarrelRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_  = ecalIsoBarrelRadiusB_[1];
    photonEcalRecHitEtaSliceB_         = ecalIsoBarrelRadiusB_[2];
    photonEcalRecHitThreshEB_          = ecalIsoBarrelRadiusB_[3];
    photonEcalRecHitThreshEtB_         = ecalIsoBarrelRadiusB_[4];

    photonHcalTowerConeInnerRadiusB_       = hcalIsoBarrelRadiusB_[0];
    photonHcalTowerConeOuterRadiusB_       = hcalIsoBarrelRadiusB_[1];
    photonHcalTowerThreshEB_               = hcalIsoBarrelRadiusB_[2];
    photonHcalDepth1TowerConeInnerRadiusB_ = hcalIsoBarrelRadiusB_[3];
    photonHcalDepth1TowerConeOuterRadiusB_ = hcalIsoBarrelRadiusB_[4];
    photonHcalDepth1TowerThreshEB_         = hcalIsoBarrelRadiusB_[5];
    photonHcalDepth2TowerConeInnerRadiusB_ = hcalIsoBarrelRadiusB_[6];
    photonHcalDepth2TowerConeOuterRadiusB_ = hcalIsoBarrelRadiusB_[7];
    photonHcalDepth2TowerThreshEB_         = hcalIsoBarrelRadiusB_[8];


    

  } else if 
    (detector==EcalEndcap) {

    trackConeOuterRadiusA_    = trkIsoEndcapRadiusA_[0];
    trackConeInnerRadiusA_    = trkIsoEndcapRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoEndcapRadiusA_[2];
    trackLipRadiusA_          = trkIsoEndcapRadiusA_[3];
    trackD0RadiusA_           = trkIsoEndcapRadiusA_[4];

    photonEcalRecHitConeInnerRadiusA_  = ecalIsoEndcapRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_  = ecalIsoEndcapRadiusA_[1];
    photonEcalRecHitEtaSliceA_         = ecalIsoEndcapRadiusA_[2];
    photonEcalRecHitThreshEA_          = ecalIsoEndcapRadiusA_[3];
    photonEcalRecHitThreshEtA_         = ecalIsoEndcapRadiusA_[4];

    photonHcalTowerConeInnerRadiusA_       = hcalIsoEndcapRadiusA_[0];
    photonHcalTowerConeOuterRadiusA_       = hcalIsoEndcapRadiusA_[1];
    photonHcalTowerThreshEA_               = hcalIsoEndcapRadiusA_[2];
    photonHcalDepth1TowerConeInnerRadiusA_ = hcalIsoEndcapRadiusA_[3];
    photonHcalDepth1TowerConeOuterRadiusA_ = hcalIsoEndcapRadiusA_[4];
    photonHcalDepth1TowerThreshEA_         = hcalIsoEndcapRadiusA_[5];
    photonHcalDepth2TowerConeInnerRadiusA_ = hcalIsoEndcapRadiusA_[6];
    photonHcalDepth2TowerConeOuterRadiusA_ = hcalIsoEndcapRadiusA_[7];
    photonHcalDepth2TowerThreshEA_         = hcalIsoEndcapRadiusA_[8];


    trackConeOuterRadiusB_    = trkIsoEndcapRadiusB_[0];
    trackConeInnerRadiusB_    = trkIsoEndcapRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoEndcapRadiusB_[2];
    trackLipRadiusA_          = trkIsoEndcapRadiusA_[3];
    trackD0RadiusA_           = trkIsoEndcapRadiusA_[4];


    photonEcalRecHitConeInnerRadiusB_  = ecalIsoEndcapRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_  = ecalIsoEndcapRadiusB_[1];
    photonEcalRecHitEtaSliceB_         = ecalIsoEndcapRadiusB_[2];
    photonEcalRecHitThreshEB_          = ecalIsoEndcapRadiusB_[3];
    photonEcalRecHitThreshEtB_         = ecalIsoEndcapRadiusB_[4];

    photonHcalTowerConeInnerRadiusB_       = hcalIsoEndcapRadiusB_[0];
    photonHcalTowerConeOuterRadiusB_       = hcalIsoEndcapRadiusB_[1];
    photonHcalTowerThreshEB_               = hcalIsoEndcapRadiusB_[2];
    photonHcalDepth1TowerConeInnerRadiusB_ = hcalIsoEndcapRadiusB_[3];
    photonHcalDepth1TowerConeOuterRadiusB_ = hcalIsoEndcapRadiusB_[4];
    photonHcalDepth1TowerThreshEB_         = hcalIsoEndcapRadiusB_[5];
    photonHcalDepth2TowerConeInnerRadiusB_ = hcalIsoEndcapRadiusB_[6];
    photonHcalDepth2TowerConeOuterRadiusB_ = hcalIsoEndcapRadiusB_[7];
    photonHcalDepth2TowerThreshEB_         = hcalIsoEndcapRadiusB_[8];

  }


  //Calculate hollow cone track isolation, CONE A
  int ntrkA=0;
  double trkisoA=0;
  calculateTrackIso(pho, e, trkisoA, ntrkA, isolationtrackThresholdA_,    
		    trackConeOuterRadiusA_, trackConeInnerRadiusA_);

  //Calculate solid cone track isolation, CONE A
  int sntrkA=0;
  double strkisoA=0;
  calculateTrackIso(pho, e, strkisoA, sntrkA, isolationtrackThresholdA_,    
		    trackConeOuterRadiusA_, 0.);

  phoisolR1.nTrkHollowCone = ntrkA;
  phoisolR1.trkSumPtHollowCone = trkisoA;
  phoisolR1.nTrkSolidCone = sntrkA;
  phoisolR1.trkSumPtSolidCone = strkisoA;

  //Calculate hollow cone track isolation, CONE B
  int ntrkB=0;
  double trkisoB=0;
  calculateTrackIso(pho, e, trkisoB, ntrkB, isolationtrackThresholdB_,    
		    trackConeOuterRadiusB_, trackConeInnerRadiusB_);

  //Calculate solid cone track isolation, CONE B
  int sntrkB=0;
  double strkisoB=0;
  calculateTrackIso(pho, e, strkisoB, sntrkB, isolationtrackThresholdB_,    
		    trackConeOuterRadiusB_, 0.);

  phoisolR2.nTrkHollowCone = ntrkB;
  phoisolR2.trkSumPtHollowCone = trkisoB;
  phoisolR2.nTrkSolidCone = sntrkB;
  phoisolR2.trkSumPtSolidCone = strkisoB;

//   std::cout << "Output from solid cone track isolation: ";
//   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;
  
  double EcalRecHitIsoA = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusA_,
						photonEcalRecHitConeInnerRadiusA_,
                                                photonEcalRecHitEtaSliceA_,
						photonEcalRecHitThreshEA_,
						photonEcalRecHitThreshEtA_);
  phoisolR1.ecalRecHitSumEt = EcalRecHitIsoA;

  double EcalRecHitIsoB = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusB_,
						photonEcalRecHitConeInnerRadiusB_,
                                                photonEcalRecHitEtaSliceB_,
						photonEcalRecHitThreshEB_,
						photonEcalRecHitThreshEtB_);
  phoisolR2.ecalRecHitSumEt = EcalRecHitIsoB;

  double HcalTowerIsoA = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusA_,
					      photonHcalTowerConeInnerRadiusA_,
					      photonHcalTowerThreshEA_, -1 );
  phoisolR1.hcalTowerSumEt = HcalTowerIsoA;


  double HcalTowerIsoB = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusB_,
					      photonHcalTowerConeInnerRadiusB_,
					      photonHcalTowerThreshEB_, -1 );
  phoisolR2.hcalTowerSumEt = HcalTowerIsoB;

  //// Hcal depth1

  double HcalDepth1TowerIsoA = calculateHcalTowerIso(pho, e, es, photonHcalDepth1TowerConeOuterRadiusA_,
					      photonHcalDepth1TowerConeInnerRadiusA_,
					      photonHcalDepth1TowerThreshEA_, 1 );
  phoisolR1.hcalDepth1TowerSumEt = HcalDepth1TowerIsoA;


  double HcalDepth1TowerIsoB = calculateHcalTowerIso(pho, e, es, photonHcalDepth1TowerConeOuterRadiusB_,
					      photonHcalDepth1TowerConeInnerRadiusB_,
					      photonHcalDepth1TowerThreshEB_, 1 );
  phoisolR2.hcalDepth1TowerSumEt = HcalDepth1TowerIsoB;



  //// Hcal depth2

  double HcalDepth2TowerIsoA = calculateHcalTowerIso(pho, e, es, photonHcalDepth2TowerConeOuterRadiusA_,
					      photonHcalDepth2TowerConeInnerRadiusA_,
					      photonHcalDepth2TowerThreshEA_, 2 );
  phoisolR1.hcalDepth2TowerSumEt = HcalDepth2TowerIsoA;


  double HcalDepth2TowerIsoB = calculateHcalTowerIso(pho, e, es, photonHcalDepth2TowerConeOuterRadiusB_,
					      photonHcalDepth2TowerConeInnerRadiusB_,
					      photonHcalDepth2TowerThreshEB_, 2 );
  phoisolR2.hcalDepth2TowerSumEt = HcalDepth2TowerIsoB;






}


void PhotonIsolationCalculator::classify(const reco::Photon* photon, 
			    bool &isEBPho,
			    bool &isEEPho,
			    bool &isEBGap,
			    bool &isEEGap,
			    bool &isEBEEGap){

  //Set fiducial flags for this photon.
  double eta = photon->superCluster()->position().eta();
  double phi = photon->superCluster()->position().phi();
  double feta = fabs(eta);

  //Are you in the Ecal Endcap (EE)?
  if(feta>1.479) 
    isEEPho = true;
  else 
    isEBPho = true;

  // Set fiducial flags (isEBGap, isEEGap...

  //Are you in the gap between EE and Ecal Barrel (EB)?
  if (fabs(feta-1.479)<.1) isEBEEGap=true; 

  // Set isEBGap if photon is 
  //  in the barrel (|eta| < 1.5), and 
  //  photon is closer than "modulePhiBoundary_" (set in cfg)
  //  to a phi module/supermodule boundary (same thing)
  if (feta < 1.5) {
    if (phi < 0) phi += TMath::Pi()*2.;
    Float_t phiRelative = fmod( phi , 20*TMath::Pi()/180 ) - 10*TMath::Pi()/180;
    if ( fabs(phiRelative) < modulePhiBoundary_ ) isEBGap=true;
  }

  // Set isEBGap if photon is between specific eta values 
  // in the "moduleEtaBoundary_" variable.
  // Loop over the vector of Eta boundaries given in the config file
  bool nearEtaBoundary = false;
  for (unsigned int i=0; i < moduleEtaBoundary_.size(); i+=2) {
    // Checks to see if it's between the 0th and 1st entry, the 2nd and 3rd entry...etc
    if ( (feta > moduleEtaBoundary_[i]) && (feta < moduleEtaBoundary_[i+1]) ) {
      //std::cout << "Photon between eta " << moduleEtaBoundary_[i] << " and " << moduleEtaBoundary_[i+1] << std::endl;
      nearEtaBoundary = true;
      break;
    }
  }

  // If it's near an eta boundary and in the Barrel
  if (nearEtaBoundary) isEBGap=true;

}

void PhotonIsolationCalculator::calculateTrackIso(const reco::Photon* photon,
						  const edm::Event& e,
						  double &trkCone,
						  int &ntrkCone,
						  double pTThresh,
						  double RCone,
						  double RinnerCone, 
						  double lip, 
						  double d0){
  int counter  =0;
  double ptSum =0.;
  
  
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel(trackInputTag_,tracks);
  if(!tracks.isValid()) {
    return;
  }
  const reco::TrackCollection* trackCollection = tracks.product();
  //Photon Eta and Phi.  Hope these are correct.
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByLabel(beamSpotProducerTag_,recoBeamSpotHandle);
  vertexBeamSpot = *recoBeamSpotHandle;
  
  PhotonTkIsolation phoIso(RCone, RinnerCone, pTThresh, lip , d0, trackCollection, math::XYZPoint(vertexBeamSpot.x0(),vertexBeamSpot.y0(),vertexBeamSpot.z0()));
  counter = phoIso.getNumberTracks(photon);
  ptSum = phoIso.getPtTracks(photon);
  //delete phoIso;
  
  ntrkCone = counter;
  trkCone = ptSum;
}



double PhotonIsolationCalculator::calculateEcalRecHitIso(const reco::Photon* photon,
					    const edm::Event& iEvent,
					    const edm::EventSetup& iSetup,
					    double RCone,
					    double RConeInner,
                                            double etaSlice,
					    double eMin,
					    double etMin){


  edm::Handle<EcalRecHitCollection> ecalhitsCollH;

  double peta = photon->superCluster()->position().eta();
  if (fabs(peta) > 1.479){
    iEvent.getByLabel(endcapecalProducer_,endcapecalCollection_, ecalhitsCollH);
  }
  else{
    iEvent.getByLabel(barrelecalProducer_,barrelecalCollection_, ecalhitsCollH);
  }
  const EcalRecHitCollection* rechitsCollection_ = ecalhitsCollH.product();

  std::auto_ptr<CaloRecHitMetaCollectionV> RecHits(0); 
  RecHits = std::auto_ptr<CaloRecHitMetaCollectionV>(new EcalRecHitMetaCollection(*rechitsCollection_));

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  double ecalIsol=0.;


  EgammaRecHitIsolation phoIso(RCone,
			       RConeInner,
                               etaSlice,
			       etMin,
			       eMin,
			       geoHandle,
			       &(*RecHits),
			       DetId::Ecal);
  ecalIsol = phoIso.getEtSum(photon);
  //  delete phoIso;

  return ecalIsol;
  

}

double PhotonIsolationCalculator::calculateHcalTowerIso(const reco::Photon* photon,
							const edm::Event& iEvent,
							const edm::EventSetup& iSetup,
							double RCone,
							double RConeInner,
							double eMin,
							signed int depth )
{

  edm::Handle<CaloTowerCollection> hcalhitsCollH;
 
  iEvent.getByLabel(hcalProducer_,hcalCollection_, hcalhitsCollH);
  
  const CaloTowerCollection *toww = hcalhitsCollH.product();

  double ecalIsol=0.;
  
  //std::cout << "before iso call" << std::endl;
  EgammaTowerIsolation phoIso(RCone,
			      RConeInner,
			      eMin,depth,
			      toww);
  ecalIsol = phoIso.getTowerEtSum(photon);
  //  delete phoIso;
  //std::cout << "after call" << std::endl;
  return ecalIsol;
  

}




