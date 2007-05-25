// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTHcalIsolation
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Monica Vazquez Acosta - CERN
//         Created:  Tue Jun 13 12:16:41 CEST 2006
// $Id: EgammaHLTHcalIsolation.cc,v 1.2 2006/10/24 10:57:40 monicava Exp $
//

// include files

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"


#define PI 3.141592654
#define TWOPI 6.283185308



float EgammaHLTHcalIsolation::isolPtSum(const reco::RecoCandidate* recocandidate, const HBHERecHitCollection* hbhe, const HFRecHitCollection* hf, const CaloGeometry* geometry){

  float hcalIsol=0.;

  float candSCphi = recocandidate->superCluster()->phi();
  float candSCeta = recocandidate->superCluster()->eta();
  if(candSCphi<0) candSCphi+=TWOPI;
  float conesizeSquared=conesize*conesize;

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
	if(newDelta<conesizeSquared) hcalIsol+=HcalHit_pth;
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
	if(newDelta<conesizeSquared) hcalIsol+=HcalHit_pth;
      }
    }      


    /*
    double HcalHit_energy=hfItr->energy();
    double HcalHit_eta=geometry->getPosition(hfItr->id()).eta();
    double HcalHit_phi=geometry->getPosition(hfItr->id()).phi();
    float HcalHit_pth=HcalHit_energy*sin(2*atan(exp(-HcalHit_eta)));
    if(HcalHit_pth>ptMin) {
      float deltaphi;
      if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
      if(candSCphi<0) candSCphi+=TWOPI;
      deltaphi=fabs(HcalHit_phi-candSCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(HcalHit_eta-candSCeta);
      float newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      if(newDelta<conesize) hcalIsol+=HcalHit_pth;
    }
    */


  }

  return hcalIsol;
  
}

