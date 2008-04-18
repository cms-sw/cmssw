#include "RecoEgamma/PhotonIdentification/interface/PhotonIDAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include <string>
#include <TMath.h>

void PhotonIDAlgo::baseSetup(const edm::ParameterSet& conf) {


  trackInputTag_ = conf.getParameter<edm::InputTag>("trackProducer");

  barrelislandsuperclusterCollection_ = conf.getParameter<std::string>("barrelislandsuperclusterCollection");
  barrelislandsuperclusterProducer_ = conf.getParameter<std::string>("barrelislandsuperclusterProducer");

  endcapSuperClusterProducer_ = conf.getParameter<std::string>("endcapSuperClustersProducer");      
  endcapsuperclusterCollection_ = conf.getParameter<std::string>("endcapsuperclusterCollection");

  barrelbasicclusterCollection_ = conf.getParameter<std::string>("barrelbasiccluterCollection");
  barrelbasicclusterProducer_ = conf.getParameter<std::string>("barrelbasicclusterProducer");
  endcapbasicclusterCollection_ = conf.getParameter<std::string>("endcapbasicclusterCollection");
  endcapbasicclusterProducer_ = conf.getParameter<std::string>("endcapbasicclusterProducer");
  
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

  


  //Track isolation calculator goes here.
  //Not my code, I stole it from:
  //RecoEgamma/EgammaIsolationAlgos/src/ElectronTkIsolation.
  //and hacked it to my own purposes.  Therefore, consider mistakes mine (AA).
  
  //Take what hopefully is the generic track collection.  Take a cone about the supercluster
  //at DCA, and sum up tracks which have dR > RinnerCone and dR < RCone.  Keep count of how many tracks
  //as well.
  int counter  =0;
  double ptSum =0.;
  

  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel(trackInputTag_,tracks);
  const reco::TrackCollection* trackCollection = tracks.product();
  //Photon Eta and Phi.  Hope these are correct.


  PhotonTkIsolation *phoIso = new PhotonTkIsolation(RCone, RinnerCone, pTThresh, 999., trackCollection);
  counter = phoIso->getNumberTracks(photon);
  ptSum = phoIso->getPtTracks(photon);
  delete phoIso;
         
  ntrkCone = counter;
  trkCone = ptSum;
}

				     

double PhotonIDAlgo::calculateBasicClusterIso(const reco::Photon* photon,
					    const edm::Event& iEvent,
					    double RCone,
					    double RConeInner,
					    double etMin){
  //This is not my code, I stole this almost entirely from 
  //RecoEgamma/EgammaIsolationAlgos/src/EgammaEcalIsolation.
  //Any mistakes in adaptation for here are mine (AA).

  //This gets a little convoluted:
  //Calculate the isolation for a hybrid supercluster.  The issue is that
  //we're going to calculate isolation using island basic clusters.

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
  EgammaEcalIsolation *phoIso = new EgammaEcalIsolation(RCone,etMin, basicClusterCollection_, islandSuperClusterCollection_);
  ecalIsol = phoIso->getEcalEtSum(photon);
  delete phoIso;

  return ecalIsol;
  

}

bool PhotonIDAlgo::isAlsoElectron(const reco::Photon* photon,
				  const edm::Event& e){

  //Get MY supercluster position
  reco::SuperClusterRef sc = photon->superCluster();
  math::XYZVector position(sc.get()->position().x(),
			   sc.get()->position().y(),
			   sc.get()->position().z());

  //get the Gsf electrons:
  edm::Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  e.getByLabel(gsfRecoInputTag_, pElectrons);
  const reco::PixelMatchGsfElectronCollection *elec = pElectrons.product();
  for(reco::PixelMatchGsfElectronCollection::const_iterator gItr = elec->begin(); gItr != elec->end(); ++gItr){
    reco::SuperClusterRef *clussy = &(*gItr).superCluster();
    const reco::SuperCluster* supercluster = clussy->get(); 
    
    math::XYZVector currentPosition(supercluster->position().x(),
				    supercluster->position().y(),
				    supercluster->position().z());
    
    double trEta = currentPosition.eta();
    double trPhi = currentPosition.phi();
    double peta = position.eta();
    double pphi = position.phi();
    double deta2 = (trEta-peta)*(trEta-peta);
    double dphi = fabs(trPhi-pphi);
    if (dphi > TMath::Pi()) dphi = TMath::Pi()*2 - dphi;
    double dphi2 = dphi*dphi;
    double dr = sqrt(deta2 + dphi2);
    if (dr < 0.0001) {
      return true;
    }
  }    
    
  return false;
}
