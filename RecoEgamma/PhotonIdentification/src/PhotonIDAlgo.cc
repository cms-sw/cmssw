#include "RecoEgamma/PhotonIdentification/interface/PhotonIDAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include <string>
#include <TMath.h>

void PhotonIDAlgo::baseSetup(const edm::ParameterSet& conf) {


  trackInputTag_ = conf.getParameter<edm::InputTag>("trackProducer");

  barrelislandsuperclusterCollection_ = conf.getParameter<std::string>("barrelislandsuperclusterCollection");
  barrelislandsuperclusterProducer_ = conf.getParameter<std::string>("barrelislandsuperclusterCollection");

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
  
  //Photon Eta and Phi.  Hope these are correct.
  double peta = photon->p4().Eta();
  double pphi = photon->p4().Phi();
  
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel(trackInputTag_,tracks);
  const reco::TrackCollection* trackCollection = tracks.product();
  
  for ( reco::TrackCollection::const_iterator itrTr  = (*trackCollection).begin(); 
	itrTr != (*trackCollection).end(); 
	++itrTr){
    math::XYZVector tmpTrackMomentumAtVtx = (*itrTr).momentum(); 
    double this_pt  = (*itrTr).pt();
    if ( this_pt < pTThresh ) 
      continue;  
    
    //This is vertex checking, I'll need to substitute PV somehow...
    //	if (fabs( (*itrTr).dz() - (*tmpTrack).dz() ) > lip_ )
    //  continue ;
    double trEta = (*itrTr).eta();
    double trPhi = (*itrTr).phi();
    double deta2 = (trEta-peta)*(trEta-peta);
    double dphi = fabs(trPhi-pphi);
    if (dphi > TMath::Pi()) dphi = TMath::Pi()*2 - dphi;
    double dphi2 = dphi*dphi;
    double dr = sqrt(deta2 + dphi2);
    if ( dr < RCone && dr > RinnerCone){
      ++counter;
      ptSum += this_pt;
    }//In cone? 
  }//end loop over tracks                 
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
  //Get MY supercluster position
  reco::SuperClusterRef sc = photon->superCluster();
  math::XYZVector position(sc.get()->position().x(),
			   sc.get()->position().y(),
			   sc.get()->position().z());
  
  // match the photon hybrid supercluster with those with Algo==0 (island)
  //Since this code doesn't use the merged collections, the Algo checking doesn't do anything.  I've
  //left it here since it is harmless:  all clusters should pass requirement, since I specifically got
  //those collections. ---A. A.

  double delta1=1000.;
  const reco::SuperCluster *matchedsupercluster=0;
  bool MATCHEDSC = false;
  
  for(reco::SuperClusterCollection::const_iterator scItr = islandSuperClusterCollection_->begin(); scItr != islandSuperClusterCollection_->end(); ++scItr){
     
    const reco::SuperCluster *supercluster = &(*scItr);
    
    math::XYZVector currentPosition(supercluster->position().x(),
				    supercluster->position().y(),
				    supercluster->position().z());
  
     
    if(supercluster->seed()->algo() == 0){    
      double trEta = currentPosition.eta();
      double trPhi = currentPosition.phi();
      double peta = position.eta();
      double pphi = position.phi();
      double deta2 = (trEta-peta)*(trEta-peta);
      double dphi = fabs(trPhi-pphi);
      if (dphi > TMath::Pi()) dphi = TMath::Pi()*2 - dphi;
      double dphi2 = dphi*dphi;
      double dr = sqrt(deta2 + dphi2);
      if (dr < delta1) {
	delta1=dr;
	matchedsupercluster = supercluster;
	MATCHEDSC = true;
      }
    }
  }
 

  //Okay, now I've made the association between my HybridSupercluster and an IslandSuperCluster.
  const reco::BasicCluster *cluster= 0;
  
  //loop over basic clusters
  for(reco::BasicClusterCollection::const_iterator cItr = basicClusterCollection_->begin(); cItr != basicClusterCollection_->end(); ++cItr){
    
    cluster = &(*cItr);
    double ebc_bcchi2 = cluster->chi2();
    int   ebc_bcalgo = cluster->algo();
    double ebc_bce    = cluster->energy();
    double ebc_bceta  = cluster->eta();
    double ebc_bcet   = ebc_bce*sin(2*atan(exp(ebc_bceta)));
    double newDelta = 0.;
 
 
    if (ebc_bcet > etMin && ebc_bcalgo == 0) {
      if (ebc_bcchi2 < 30.) {
	
	if(MATCHEDSC){
	  bool inSuperCluster = false;
	  
	  reco::basicCluster_iterator theEclust = matchedsupercluster->clustersBegin();
	  // loop over the basic clusters of the matched supercluster

	  //I consider this somewhat wacky, if you are a basiccluster which was included in my
	  //matched island supercluster, then you don't count against me for isolation.  If you AREN'T
	  //included in my supercluster, then you are assumed to be from something else.  I think this
	  //will have to be eliminated, especially if we're going to use fixed arrays for photons.
	  for(;theEclust != matchedsupercluster->clustersEnd();
	      theEclust++) {
	    if ((**theEclust) ==  (*cluster) ) inSuperCluster = true;
	  }
	  if (!inSuperCluster) {
	    
	    math::XYZVector basicClusterPosition(cluster->position().x(),
						 cluster->position().y(),
						 cluster->position().z());
	    double trEta = basicClusterPosition.eta();
	    double trPhi = basicClusterPosition.phi();
	    double peta = position.eta();
	    double pphi = position.phi();
	    double deta2 = (trEta-peta)*(trEta-peta);
	    double dphi = fabs(trPhi-pphi);
	    if (dphi > TMath::Pi()) dphi = TMath::Pi()*2 - dphi;
	    double dphi2 = dphi*dphi;
	    double dr = sqrt(deta2 + dphi2);
	    if(dr < RCone
	       && newDelta > RConeInner) {
	      ecalIsol+=ebc_bcet;
	    }
	  }
	}
      } // matches ebc_bcchi2
    } // matches ebc_bcet && ebc_bcalgo
    
  }
  
  //  std::cout << "Will return ecalIsol = " << ecalIsol << std::endl; 
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
