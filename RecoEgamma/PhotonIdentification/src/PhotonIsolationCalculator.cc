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

  barrelecalCollection_ = conf.getParameter<std::string>("barrelEcalRecHitCollection");
  barrelecalProducer_ = conf.getParameter<std::string>("barrelEcalRecHitProducer");
  endcapecalCollection_ = conf.getParameter<std::string>("endcapEcalRecHitCollection");
  endcapecalProducer_ = conf.getParameter<std::string>("endcapEcalRecHitProducer");
  hcalCollection_ = conf.getParameter<std::string>("HcalRecHitCollection");
  hcalProducer_ = conf.getParameter<std::string>("HcalRecHitProducer");

  //  gsfRecoInputTag_ = conf.getParameter<edm::InputTag>("GsfRecoCollection");
  modulePhiBoundary_ = conf.getParameter<double>("modulePhiBoundary");
  moduleEtaBoundary_ = conf.getParameter<std::vector<double> >("moduleEtaBoundary");

  trackConeOuterRadiusA_ = conf.getParameter<double>("TrackConeOuterRadiusA");
  trackConeInnerRadiusA_ = conf.getParameter<double>("TrackConeInnerRadiusA");
  isolationtrackThresholdA_ = conf.getParameter<double>("isolationtrackThresholdA");

  photonEcalRecHitConeInnerRadiusA_ = conf.getParameter<double>("EcalRecHitInnerRadiusA");
  photonEcalRecHitConeOuterRadiusA_ = conf.getParameter<double>("EcalRecHitOuterRadiusA");
  photonEcalRecHitEtaSliceA_ = conf.getParameter<double>("EcalRecHitEtaSliceA");
  photonEcalRecHitThreshEA_ = conf.getParameter<double>("EcalRecThreshEA");
  photonEcalRecHitThreshEtA_ = conf.getParameter<double>("EcalRecThreshEtA");

  photonHcalTowerConeInnerRadiusA_ = conf.getParameter<double>("HcalTowerInnerRadiusA");
  photonHcalTowerConeOuterRadiusA_ = conf.getParameter<double>("HcalTowerOuterRadiusA");
  photonHcalTowerThreshEA_ = conf.getParameter<double>("HcalTowerThreshEA");

  photonHcalDepth1TowerConeInnerRadiusA_ = conf.getParameter<double>("HcalDepth1TowerInnerRadiusA");
  photonHcalDepth1TowerConeOuterRadiusA_ = conf.getParameter<double>("HcalDepth1TowerOuterRadiusA");
  photonHcalDepth1TowerThreshEA_ = conf.getParameter<double>("HcalDepth1TowerThreshEA");

  photonHcalDepth2TowerConeInnerRadiusA_ = conf.getParameter<double>("HcalDepth2TowerInnerRadiusA");
  photonHcalDepth2TowerConeOuterRadiusA_ = conf.getParameter<double>("HcalDepth2TowerOuterRadiusA");
  photonHcalDepth2TowerThreshEA_ = conf.getParameter<double>("HcalDepth2TowerThreshEA");

  trackConeOuterRadiusB_ = conf.getParameter<double>("TrackConeOuterRadiusB");
  trackConeInnerRadiusB_ = conf.getParameter<double>("TrackConeInnerRadiusB");
  isolationtrackThresholdB_ = conf.getParameter<double>("isolationtrackThresholdB");

  photonEcalRecHitConeInnerRadiusB_ = conf.getParameter<double>("EcalRecHitInnerRadiusB");
  photonEcalRecHitConeOuterRadiusB_ = conf.getParameter<double>("EcalRecHitOuterRadiusB");
  photonEcalRecHitEtaSliceB_ = conf.getParameter<double>("EcalRecHitEtaSliceB");
  photonEcalRecHitThreshEB_ = conf.getParameter<double>("EcalRecThreshEB");
  photonEcalRecHitThreshEtB_ = conf.getParameter<double>("EcalRecThreshEtB");

  photonHcalTowerConeInnerRadiusB_ = conf.getParameter<double>("HcalTowerInnerRadiusB");
  photonHcalTowerConeOuterRadiusB_ = conf.getParameter<double>("HcalTowerOuterRadiusB");
  photonHcalTowerThreshEB_ = conf.getParameter<double>("HcalTowerThreshEB");


  photonHcalDepth1TowerConeInnerRadiusB_ = conf.getParameter<double>("HcalDepth1TowerInnerRadiusB");
  photonHcalDepth1TowerConeOuterRadiusB_ = conf.getParameter<double>("HcalDepth1TowerOuterRadiusB");
  photonHcalDepth1TowerThreshEB_ = conf.getParameter<double>("HcalDepth1TowerThreshEB");

  photonHcalDepth2TowerConeInnerRadiusB_ = conf.getParameter<double>("HcalDepth2TowerInnerRadiusB");
  photonHcalDepth2TowerConeOuterRadiusB_ = conf.getParameter<double>("HcalDepth2TowerOuterRadiusB");
  photonHcalDepth2TowerThreshEB_ = conf.getParameter<double>("HcalDepth2TowerThreshEB");





}


void PhotonIsolationCalculator::calculate(const reco::Photon* pho,
				     const edm::Event& e,
				     const edm::EventSetup& es,
				     reco::Photon::FiducialFlags& phofid, 
				     reco::Photon::IsolationVariables& phoisolR1, 
				     reco::Photon::IsolationVariables& phoisolR2){


  //Get fiducial information
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
				     double RinnerCone){
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

  
  PhotonTkIsolation phoIso(RCone, RinnerCone, pTThresh, 2.,2000., trackCollection, math::XYZPoint(0,0,0));
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




