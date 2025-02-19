// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTHcalIsolationDoubleCone
// 
// Implementation:
//     use double cone to exclude shower leakage
//     mostly identical to EgammaHLTHcalIsolation, but
//     with an inner exclusion cone 
// Original Author:  
//         Created:  Tue Jun 13 12:16:41 CEST 2006
// $Id: EgammaHLTHcalIsolationDoubleCone.cc,v 1.1 2007/05/31 19:38:50 mpieri Exp $
//

// include files

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolationDoubleCone.h"


#define PI 3.141592654
#define TWOPI 6.283185308



float EgammaHLTHcalIsolationDoubleCone::isolPtSum(const reco::RecoCandidate* recocandidate, const HBHERecHitCollection* hbhe, const HFRecHitCollection* hf, const CaloGeometry* geometry){

  float hcalIsol=0.;

  float candSCphi = recocandidate->superCluster()->phi();
  float candSCeta = recocandidate->superCluster()->eta();
  if(candSCphi<0) candSCphi+=TWOPI;
  float conesizeSquared=conesize*conesize;
  float exclusionSquared= exclusion*exclusion;

  for(HBHERecHitCollection::const_iterator hbheItr = hbhe->begin(); hbheItr != hbhe->end(); ++hbheItr){
    double HcalHit_eta=geometry->getPosition(hbheItr->id()).eta(); //Attention getpos
    if(fabs(HcalHit_eta-candSCeta)<conesize) {
      float HcalHit_pth=hbheItr->energy()*sin(2*atan(exp(-HcalHit_eta)));
      if(HcalHit_pth>ptMin) {
	double HcalHit_phi=geometry->getPosition(hbheItr->id()).phi();
	float deltaeta=fabs(HcalHit_eta-candSCeta);
	if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
	float deltaphi=fabs(HcalHit_phi-candSCphi);
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
	float newDelta= (deltaphi*deltaphi+ deltaeta*deltaeta);
	if(newDelta<conesizeSquared && newDelta>exclusionSquared ) hcalIsol+=HcalHit_pth;
      }
    }      
  }

  for(HFRecHitCollection::const_iterator hfItr = hf->begin(); hfItr != hf->end(); ++hfItr){
    double HcalHit_eta=geometry->getPosition(hfItr->id()).eta(); //Attention getpos
    if(fabs(HcalHit_eta-candSCeta)<conesize) {
      float HcalHit_pth=hfItr->energy()*sin(2*atan(exp(-HcalHit_eta)));
      if(HcalHit_pth>ptMin) {
	double HcalHit_phi=geometry->getPosition(hfItr->id()).phi();
	float deltaeta=fabs(HcalHit_eta-candSCeta);
	float deltaphi;
	if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
	if(candSCphi<0) candSCphi+=TWOPI;
	deltaphi=fabs(HcalHit_phi-candSCphi);
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
	float newDelta= (deltaphi*deltaphi+ deltaeta*deltaeta);
	if(newDelta<conesizeSquared && newDelta>exclusionSquared ) hcalIsol+=HcalHit_pth;
      }
    }      
  }


  return hcalIsol;
  
}

