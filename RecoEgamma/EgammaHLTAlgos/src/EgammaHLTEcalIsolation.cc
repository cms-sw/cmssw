// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTEcalIsolation
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 12:16:00 CEST 2006
// $Id: EgammaHLTEcalIsolation.cc,v 1.6 2013/05/30 21:48:56 gartung Exp $
//

// system include files

// user include files
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h"

#define PI 3.141592654
#define TWOPI 6.283185308


float EgammaHLTEcalIsolation::isolPtSum(const reco::RecoCandidate* recocandidate, 
					const std::vector<const reco::SuperCluster*>& sclusters,
					const std::vector<const reco::BasicCluster*>& bclusters){

  float ecalIsol=0.;
  
  float candSCphi = recocandidate->superCluster()->phi();
  float candSCeta = recocandidate->superCluster()->eta();

  
  // match the photon hybrid supercluster with those with Algo==0 (island)
  float delta1=1000.;
  float deltacur=1000.;
  const reco::SuperCluster *matchedsupercluster=0;
  bool MATCHEDSC = false;



  for(std::vector<const reco::SuperCluster*>::const_iterator scItr = sclusters.begin(); scItr != sclusters.end(); ++scItr){
    
    
    const reco::SuperCluster *supercluster = *scItr;
    
    float SCphi = supercluster->phi();
    float SCeta = supercluster->eta();
   
    if(supercluster->seed()->algo() == algoType_){
      float deltaphi;
      if(candSCphi<0) candSCphi+=TWOPI;
      if(SCphi<0) SCphi+=TWOPI;
      deltaphi=fabs(candSCphi-SCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(SCeta-candSCeta);
      deltacur = sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      
      if (deltacur < delta1) {
        delta1=deltacur;
	matchedsupercluster = supercluster;
	MATCHEDSC = true;
      }
    }
  }

  const reco::BasicCluster *cluster= 0;

  //loop over basic clusters
  for(std::vector<const reco::BasicCluster*>::const_iterator cItr = bclusters.begin(); cItr != bclusters.end(); ++cItr){
 
    cluster = *cItr;
//    float ebc_bcchi2 = cluster->chi2(); //chi2 for SC was useless and it is removed in 31x
    int   ebc_bcalgo = cluster->algo();
    float ebc_bce    = cluster->energy();
    float ebc_bceta  = cluster->eta();
    float ebc_bcphi  = cluster->phi();
    float ebc_bcet   = ebc_bce*sin(2*atan(exp(ebc_bceta)));
    float newDelta;


    if (ebc_bcet > etMin && ebc_bcalgo == algoType_ ) {
      //  if (ebc_bcchi2 < 30.) {
	
	if(MATCHEDSC){
	  bool inSuperCluster = false;

	  reco::CaloCluster_iterator theEclust = matchedsupercluster->clustersBegin();
	  // loop over the basic clusters of the matched supercluster
	  for(;theEclust != matchedsupercluster->clustersEnd();
	      theEclust++) {
	    if (&(**theEclust) ==  cluster) inSuperCluster = true;
	  }
	  if (!inSuperCluster) {
	    float deltaphi;
	    if(ebc_bcphi<0) ebc_bcphi+=TWOPI;
	    if(candSCphi<0) candSCphi+=TWOPI;
	    deltaphi=fabs(ebc_bcphi-candSCphi);
	    if(deltaphi>TWOPI) deltaphi-=TWOPI;
	    if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
	    float deltaeta=fabs(ebc_bceta-candSCeta);
	    newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
	    if(newDelta < conesize) {
	      ecalIsol+=ebc_bcet;
	    }
	  }
	}
	//  } // matches ebc_bcchi2
    } // matches ebc_bcet && ebc_bcalgo

  }


 return ecalIsol;

}
