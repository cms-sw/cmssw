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
// $Id$
//

// system include files

// user include files
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h"

#define PI 3.141592654
#define TWOPI 6.283185308


float EgammaHLTEcalIsolation::photonPtSum(const reco::Photon *photon, const reco::SuperClusterCollection& sclusters
					  , const reco::BasicClusterCollection& bclusters
					  ){
  float ecalIsol=0.;
  
  float phoSCphi = photon->superCluster()->phi();
  float phoSCeta = photon->superCluster()->eta();
  
  // match the photon hybrid supercluster with those with Algo==1
  float delta1=1000.;
  float deltacur=1000.;
  const reco::SuperCluster *matchedsupercluster=0;
  bool MATCHEDSC = false;
  
  
  for(reco::SuperClusterCollection::const_iterator scItr = sclusters.begin(); scItr != sclusters.end(); ++scItr){
    
    
    const reco::SuperCluster *supercluster = &(*scItr);
    
    float SCphi = supercluster->phi();
    float SCeta = supercluster->eta();
    
    if(supercluster->seed()->algo() == 1){
      float deltaphi;
      if(phoSCphi<0) phoSCphi+=TWOPI;
      if(SCphi<0) SCphi+=TWOPI;
      deltaphi=fabs(phoSCphi-SCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(SCeta-phoSCeta);
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
  for(reco::BasicClusterCollection::const_iterator cItr = bclusters.begin(); cItr != bclusters.end(); ++cItr){
 
    cluster = &(*cItr);
    float ebc_bcchi2 = cluster->chi2();
    int   ebc_bcalgo = cluster->algo();
    float ebc_bce    = cluster->energy();
    float ebc_bceta  = cluster->eta();
    float ebc_bcphi  = cluster->phi();
    float ebc_bcet   = ebc_bce*sin(2*atan(exp(ebc_bceta)));
    float newDelta;



    if (ebc_bcet > etMinG && ebc_bcalgo == 1) {
      if (ebc_bcchi2 < 30.) {
	
	if(MATCHEDSC){
	  bool inSuperCluster = false;


	  reco::basicCluster_iterator theEclust = matchedsupercluster->clustersBegin();

	  for(;theEclust != matchedsupercluster->clustersEnd();
	      theEclust++) {
	    if (&(**theEclust) ==  cluster) inSuperCluster = true;
	  }
	  if (!inSuperCluster) {
	    float deltaphi;
	    if(ebc_bcphi<0) ebc_bcphi+=TWOPI;
	    if(phoSCphi<0) phoSCphi+=TWOPI;
	    deltaphi=fabs(ebc_bcphi-phoSCphi);
	    if(deltaphi>TWOPI) deltaphi-=TWOPI;
	    if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
	    float deltaeta=fabs(ebc_bceta-phoSCeta);
	    newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
	    if(newDelta < conesizeG) {
	      ecalIsol+=ebc_bcet;
	    }
	  }
	}
      } // matches ebc_bcchi2
    } // matches ebc_bcet && ebc_bcalgo

  }


 return ecalIsol;

}
