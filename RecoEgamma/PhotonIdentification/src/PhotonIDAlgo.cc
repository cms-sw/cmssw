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
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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


}



void PhotonIDAlgo::classify(const reco::Photon* photon, 
			    bool &isEBPho,
			    bool &isEEPho,
			    bool &isEBGap,
			    bool &isEEGap,
			    bool &isEBEEGap){

  //Set fiducial flags for this photon.
  double eta = photon->p4().Eta();
  double phi = photon->p4().Phi();
  double feta = fabs(eta);

  //Are you in the EE?
  if(feta>1.479) 
    isEEPho = true;
  else 
    isEBPho = true;

  //Are you in the gap between EE and EB?
  if (fabs(feta-1.479)<.1) isEBEEGap=true; 
  
  
  //fiducial cuts, currently only for EB, since I don't know
  //EE yet.

  //Module boundaries in phi (supermodule boundaries):
  float phigap = fabs(phi-int(phi*9/3.1416)*3.1416/9.);
  if(phigap > 1.65 && phigap <1.85) isEBGap=true;

  //Module boundaries in eta (supercrystal boundaries):
  if(fabs(eta)<.05) isEBGap=true;
  if(fabs(eta)>.4 && fabs(eta)<.5) isEBGap=true;
  if(fabs(eta)>.75 && fabs(eta)<.85) isEBGap=true;
  if(fabs(eta)>1.1 && fabs(eta)<1.2) isEBGap=true;
  if(fabs(eta)>1.43) isEBGap=true;
  
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
  const reco::TrackCollection* trackCollection = tracks.product();
  //Photon Eta and Phi.  Hope these are correct.
  
  
  PhotonTkIsolation phoIso(RCone, RinnerCone, pTThresh, 999., trackCollection);
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
  edm::Handle<reco::SuperClusterCollection> superIslandClusterH;

  double peta = photon->p4().Eta();
  if (fabs(peta) > 1.479){
    iEvent.getByLabel(endcapbasicclusterProducer_,endcapbasicclusterCollection_,basicClusterH);
    iEvent.getByLabel(endcapSuperClusterProducer_,endcapsuperclusterCollection_,superIslandClusterH);
  }
  else{
    iEvent.getByLabel(barrelbasicclusterProducer_,barrelbasicclusterCollection_,basicClusterH);
    iEvent.getByLabel(barrelislandsuperclusterProducer_,barrelislandsuperclusterCollection_,superIslandClusterH);
  }
  const reco::BasicClusterCollection* basicClusterCollection_ = basicClusterH.product();
  const reco::SuperClusterCollection* islandSuperClusterCollection_ = superIslandClusterH.product();

  double ecalIsol=0.;
  EgammaEcalIsolation phoIso(RCone,etMin, basicClusterCollection_, islandSuperClusterCollection_);
  ecalIsol = phoIso.getEcalEtSum(photon);
  //  delete phoIso;

  return ecalIsol;
  

}

double PhotonIDAlgo::calculateEcalRecHitIso(const reco::Photon* photon,
					    const edm::Event& iEvent,
					    const edm::EventSetup& iSetup,
					    double RCone,
					    double RConeInner,
					    double etMin){


  edm::Handle<EcalRecHitCollection> ecalhitsCollH;

  double peta = photon->p4().Eta();
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
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
  double ecalIsol=0.;
  
  EgammaRecHitIsolation phoIso(RCone,
			       RConeInner,
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
  double peta = photon->p4().Eta();
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
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
					    double etMin){


  edm::Handle<HBHERecHitCollection> hcalhitsCollH;
 
  iEvent.getByLabel(hcalProducer_,hcalCollection_, hcalhitsCollH);

  const HBHERecHitCollection* rechitsCollection_ = hcalhitsCollH.product();

  std::auto_ptr<CaloRecHitMetaCollectionV> RecHits(0); 
  RecHits = std::auto_ptr<CaloRecHitMetaCollectionV>(new HBHERecHitMetaCollection(*rechitsCollection_));

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
  double ecalIsol=0.;
  
  EgammaRecHitIsolation phoIso(RCone,
			       RConeInner,
			       etMin,
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
