/** \class PhotonIDAlgo
 *  Determine and Set quality information on Photon Objects
 *
 *  \author A. Askew, N. Marinelli, M.B. Anderson
 */

#include "RecoEgamma/PhotonIdentification/interface/PhotonIDAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include <string>
#include <TMath.h>


void PhotonIDAlgo::baseSetup(const edm::ParameterSet& conf) {


  trackInputTag_ = conf.getParameter<edm::InputTag>("trackProducer");

  barrelecalCollection_ = conf.getParameter<std::string>("barrelEcalRecHitCollection");
  barrelecalProducer_ = conf.getParameter<std::string>("barrelEcalRecHitProducer");
  endcapecalCollection_ = conf.getParameter<std::string>("endcapEcalRecHitCollection");
  endcapecalProducer_ = conf.getParameter<std::string>("endcapEcalRecHitProducer");
  hcalCollection_ = conf.getParameter<std::string>("HcalRecHitCollection");
  hcalProducer_ = conf.getParameter<std::string>("HcalRecHitProducer");

  gsfRecoInputTag_ = conf.getParameter<edm::InputTag>("GsfRecoCollection");

  modulePhiBoundary_ = conf.getParameter<double>("modulePhiBoundary");
  moduleEtaBoundary_ = conf.getParameter<std::vector<double> >("moduleEtaBoundary");

}



void PhotonIDAlgo::classify(const reco::Photon* photon, 
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

  // Set isEBGap if photon is closer than "modulePhiBoundary_" (set in cfg)
  // to a phi module/supermodule boundary (same thing)
  if (phi < 0) phi += TMath::Pi()*2.;
  Float_t phiRelative = fmod( phi , 20*TMath::Pi()/180 ) - 10*TMath::Pi()/180;
  if ( fabs(phiRelative) < modulePhiBoundary_ ) isEBGap=true;

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

void PhotonIDAlgo::calculateTrackIso(const reco::Photon* photon,
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

  
  PhotonTkIsolation phoIso(RCone, RinnerCone, pTThresh, 2., trackCollection);
  counter = phoIso.getNumberTracks(photon);
  ptSum = phoIso.getPtTracks(photon);
  //delete phoIso;
  
  ntrkCone = counter;
  trkCone = ptSum;
}



double PhotonIDAlgo::calculateBasicClusterIso(const reco::Photon* photon,
					      const edm::Event& iEvent,
					      double RCone,
					      double RConeInner,
					      double etMin)
{
					      

  edm::Handle<reco::BasicClusterCollection> basicClusterH;
  edm::Handle<reco::SuperClusterCollection> endcapSuperClusterH;

  double peta = photon->p4().Eta();
  if (fabs(peta) > 1.479){
    iEvent.getByLabel(endcapbasicclusterProducer_,endcapbasicclusterCollection_,basicClusterH);
    iEvent.getByLabel(endcapSuperClusterProducer_,endcapsuperclusterCollection_,endcapSuperClusterH);
  }
  else{
    iEvent.getByLabel(barrelbasicclusterProducer_,barrelbasicclusterCollection_,basicClusterH);
    iEvent.getByLabel(barrelsuperclusterProducer_,barrelsuperclusterCollection_,endcapSuperClusterH);
  }
  const reco::BasicClusterCollection* basicClusterCollection_ = basicClusterH.product();
  const reco::SuperClusterCollection* endcapSuperClusterCollection_ = endcapSuperClusterH.product();

  double ecalIsol=0.;
  EgammaEcalIsolation phoIso(RCone,etMin, basicClusterCollection_, endcapSuperClusterCollection_);
  ecalIsol = phoIso.getEcalEtSum(photon);
  //  delete phoIso;

  return ecalIsol;
  

}

double PhotonIDAlgo::calculateEcalRecHitIso(const reco::Photon* photon,
					    const edm::Event& iEvent,
					    const edm::EventSetup& iSetup,
					    double RCone,
					    double RConeInner,
                                            double etaSlice,
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
			       geoHandle,
			       &(*RecHits),
			       DetId::Ecal);
  ecalIsol = phoIso.getEtSum(photon);
  //  delete phoIso;

  return ecalIsol;
  

}

double PhotonIDAlgo::calculateR9(const reco::Photon* photon,
				 const edm::Event& iEvent,
				 const edm::EventSetup& iSetup
				 ){


  edm::Handle<EcalRecHitCollection> ecalhitsCollH;
  double peta = photon->superCluster()->position().eta();
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  //const CaloGeometry& geometry = *geoHandle;
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology *topology = pTopology.product();
  // const CaloSubdetectorGeometry *geometry_p;
  if (fabs(peta) > 1.479){
    iEvent.getByLabel(endcapecalProducer_,endcapecalCollection_, ecalhitsCollH);
  }
  else{
    iEvent.getByLabel(barrelecalProducer_,barrelecalCollection_, ecalhitsCollH);
  }
  const EcalRecHitCollection* rechitsCollection_ = ecalhitsCollH.product();

  reco::SuperClusterRef scref = photon->superCluster();
  const reco::SuperCluster *sc = scref.get();
  const reco::BasicClusterRef bcref = sc->seed();
  const reco::BasicCluster *bc = bcref.get();

  if (fabs(peta) > 1.479){

    float e3x3 = EcalClusterTools::e3x3(*bc, rechitsCollection_, topology);
    double r9 = e3x3 / (photon->superCluster()->rawEnergy());
    return r9;
  }					  
  else{
    float e3x3 = EcalClusterTools::e3x3(*bc, rechitsCollection_, topology);
    double r9 = e3x3 / (photon->superCluster()->rawEnergy());
    return r9;
  }

}

double PhotonIDAlgo::calculateHcalRecHitIso(const reco::Photon* photon,
					    const edm::Event& iEvent,
					    const edm::EventSetup& iSetup,
					    double RCone,
					    double RConeInner,
                                            double etaSlice,
					    double etMin){


  edm::Handle<HBHERecHitCollection> hcalhitsCollH;
 
  iEvent.getByLabel(hcalProducer_,hcalCollection_, hcalhitsCollH);

  const HBHERecHitCollection* rechitsCollection_ = hcalhitsCollH.product();

  std::auto_ptr<CaloRecHitMetaCollectionV> RecHits(0); 
  RecHits = std::auto_ptr<CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*rechitsCollection_));

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  double ecalIsol=0.;
  

  EgammaRecHitIsolation phoIso(RCone,
			       RConeInner,
			       etMin,
                               etaSlice,
			       geoHandle,
			       &(*RecHits),
			       DetId::Hcal);
  ecalIsol = phoIso.getEtSum(photon);
  //  delete phoIso;

  return ecalIsol;
  

}



bool PhotonIDAlgo::isAlsoElectron(const reco::Photon* photon,
				  const edm::Event& e){

  //Currently some instability with GsfPixelMatchElectronCollection
  //causes this simple code to die horribly.  Thus we simply return false
  //for now.

  //Get MY supercluster position
//   std::cout << "Checking isAlsoElectron code: " << std::endl;
//   reco::SuperClusterRef sc = photon->superCluster();
//   float PhoCaloE = sc.get()->energy();

//   math::XYZVector position(sc.get()->position().x(),
// 			   sc.get()->position().y(),
// 			   sc.get()->position().z());
  
//   std::cout << "Got supercluster position: Photon." << std::endl;
  //get the Gsf electrons:
//   edm::Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
//   e.getByLabel(gsfRecoInputTag_, pElectrons);
//   std::cout << "Got GsfElectronCollection: " << std::endl;
//   float PhoCaloE=0;

//   const reco::PixelMatchGsfElectronCollection *elec = pElectrons.product();
//   for(reco::PixelMatchGsfElectronCollection::const_iterator gItr = elec->begin(); gItr != elec->end(); ++gItr){

//     std::cout << "Got Electron: " << std::endl;
//     float EleCaloE = gItr->caloEnergy();
//     std::cout << "Energy: " << EleCaloE << std::endl;
//     std::cout << "Photon E: " << PhoCaloE << std::endl;
//     float dE = fabs(EleCaloE-PhoCaloE);
//     std::cout << "Made comparison. " << std::endl;

//     if(dE < 0.0001) return true;

//   }    
    
  return false;
}
