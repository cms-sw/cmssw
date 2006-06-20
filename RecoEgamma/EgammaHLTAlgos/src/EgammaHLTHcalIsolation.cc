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
// $Id$
//

// include files

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"


#define PI 3.141592654
#define TWOPI 6.283185308



float EgammaHLTHcalIsolation::electronPtSum(const reco::Electron *electron, const HBHERecHitCollection& hbhe, const HFRecHitCollection& hf, const CaloGeometry& geometry){

  float hcalIsol=0.;

  float eleSCphi = electron->superCluster()->phi();
  float eleSCeta = electron->superCluster()->eta();
  
  
  for(HBHERecHitCollection::const_iterator hbheItr = hbhe.begin(); hbheItr != hbhe.end(); ++hbheItr){
    double HcalHit_energy=hbheItr->energy();
    double HcalHit_eta=geometry.getPosition(hbheItr->id()).eta();
    double HcalHit_phi=geometry.getPosition(hbheItr->id()).phi();
    float HcalHit_pth=HcalHit_energy*sin(2*atan(exp(-HcalHit_eta)));
    if(HcalHit_pth>ptMin) {
      float deltaphi;
      if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
      if(eleSCphi<0) eleSCphi+=TWOPI;
      deltaphi=fabs(HcalHit_phi-eleSCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(HcalHit_eta-eleSCeta);
      float newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      if(newDelta<conesize) hcalIsol+=HcalHit_pth;
    }      
  }

  for(HFRecHitCollection::const_iterator hfItr = hf.begin(); hfItr != hf.end(); ++hfItr){
    double HcalHit_energy=hfItr->energy();
    double HcalHit_eta=geometry.getPosition(hfItr->id()).eta();
    double HcalHit_phi=geometry.getPosition(hfItr->id()).phi();
    float HcalHit_pth=HcalHit_energy*sin(2*atan(exp(-HcalHit_eta)));
    if(HcalHit_pth>ptMin) {
      float deltaphi;
      if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
      if(eleSCphi<0) eleSCphi+=TWOPI;
      deltaphi=fabs(HcalHit_phi-eleSCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(HcalHit_eta-eleSCeta);
      float newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      if(newDelta<conesize) hcalIsol+=HcalHit_pth;
    }
  }

  return hcalIsol;
  
}


float EgammaHLTHcalIsolation::photonPtSum(const reco::Photon *photon, const HBHERecHitCollection& hbhe, const HFRecHitCollection& hf, const CaloGeometry& geometry){

  float hcalIsol=0.;

  float phoSCphi = photon->superCluster()->phi();
  float phoSCeta = photon->superCluster()->eta();

  
  for(HBHERecHitCollection::const_iterator hbheItr = hbhe.begin(); hbheItr != hbhe.end(); ++hbheItr){
    double HcalHit_energy=hbheItr->energy();
    double HcalHit_eta=geometry.getPosition(hbheItr->id()).eta();
    double HcalHit_phi=geometry.getPosition(hbheItr->id()).phi();
    float HcalHit_pth=HcalHit_energy*sin(2*atan(exp(-HcalHit_eta)));
    if(HcalHit_pth>ptMinG) {
      float deltaphi;
      if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
      if(phoSCphi<0) phoSCphi+=TWOPI;
      deltaphi=fabs(HcalHit_phi-phoSCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(HcalHit_eta-phoSCeta);
      float newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      if(newDelta<conesizeG) hcalIsol+=HcalHit_pth;
    }      
  }

  for(HFRecHitCollection::const_iterator hfItr = hf.begin(); hfItr != hf.end(); ++hfItr){
    double HcalHit_energy=hfItr->energy();
    double HcalHit_eta=geometry.getPosition(hfItr->id()).eta();
    double HcalHit_phi=geometry.getPosition(hfItr->id()).phi();
    float HcalHit_pth=HcalHit_energy*sin(2*atan(exp(-HcalHit_eta)));
    if(HcalHit_pth>ptMinG) {
      float deltaphi;
      if(HcalHit_phi<0) HcalHit_phi+=TWOPI;
      if(phoSCphi<0) phoSCphi+=TWOPI;
      deltaphi=fabs(HcalHit_phi-phoSCphi);
      if(deltaphi>TWOPI) deltaphi-=TWOPI;
      if(deltaphi>PI) deltaphi=TWOPI-deltaphi;
      float deltaeta=fabs(HcalHit_eta-phoSCeta);
      float newDelta= sqrt(deltaphi*deltaphi+ deltaeta*deltaeta);
      if(newDelta<conesizeG) hcalIsol+=HcalHit_pth;
    }
  }

  return hcalIsol;
  
}
