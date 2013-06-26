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
// $Id: EgammaHLTHcalIsolation.cc,v 1.4 2010/08/12 15:25:02 sharper Exp $
//

// include files

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//first is the sum E, second is the sum Et
std::pair<float,float> EgammaHLTHcalIsolation::getSum(const float candEta,const float candPhi, const HBHERecHitCollection* hbhe, const CaloGeometry* geometry,const HcalSeverityLevelComputer* hcalSevLvlAlgo,const HcalChannelQuality* dbHcalChStatus)const //note last two pointers can be NULL
{
  float sumE=0.;
  float sumEt=0.;
  
  for(HBHERecHitCollection::const_iterator hbheItr = hbhe->begin(); hbheItr != hbhe->end(); ++hbheItr){
    if(passCleaning_(&(*hbheItr),hcalSevLvlAlgo,dbHcalChStatus)){
      //      if(hbheItr->id().ietaAbs()==29) continue;

      HcalDetId id = hbheItr->id();
      if(!(id.ietaAbs()==28 && id.depth()==3)){ //default normal case
	float energy = hbheItr->energy();
	const GlobalPoint& pos = geometry->getPosition(id);
	if(acceptHit_(id,pos,energy,candEta,candPhi)){
	  sumE+=energy;
	  sumEt+=energy*sin(pos.theta());
	}
      }else{
	//the special case, tower 28 depth 3 is split between tower 28 and 29 when using calo towers so we have to emulate it. To do this we need to divide energy by 2 and then check seperately if 28 and 29 are accepted
	float energy = hbheItr->energy()/2.;
	HcalDetId tower28Id(id.subdet(),28*id.zside(),id.iphi(),2);
	const GlobalPoint& tower28Pos = geometry->getPosition(tower28Id);
	if(acceptHit_(id,tower28Pos,energy,candEta,candPhi)){
	  sumE+=energy;
	  sumEt+=energy*sin(tower28Pos.theta());
	}
	HcalDetId tower29Id(id.subdet(),29*id.zside(),id.iphi(),2);
	const GlobalPoint& tower29Pos = geometry->getPosition(tower29Id);
	if(acceptHit_(id,tower29Pos,energy,candEta,candPhi)){
	  sumE+=energy;
	  sumEt+=energy*sin(tower29Pos.theta());
	}
      }//end of the special case for tower 28 depth 3
    }//end cleaning check
  }//end of loop over all rec hits
  return std::make_pair(sumE,sumEt);
  
}


//true if the hit passes Et, E, dR and depth requirements
bool EgammaHLTHcalIsolation::acceptHit_(const HcalDetId id,const GlobalPoint& pos,const float hitEnergy,const float candEta,const float candPhi)const
{
  if(passMinE_(hitEnergy,id) && passDepth_(id)){ //doing the energy and depth cuts first will avoid potentially slow eta calc
    float innerConeSq = innerCone_*innerCone_; 
    float outerConeSq = outerCone_*outerCone_;

    float hitEta = pos.eta();
    float hitPhi = pos.phi();
    
    float dR2 = reco::deltaR2(candEta,candPhi,hitEta,hitPhi);
    
    if(dR2>=innerConeSq && dR2<outerConeSq) { //pass inner and outer cone cuts
      float hitEt = hitEnergy*sin(2*atan(exp(-hitEta)));
      if(passMinEt_(hitEt,id)) return true; //and we've passed the last requirement
    }//end dR check
  }//end min energy + depth check
  
  return false;
}

bool EgammaHLTHcalIsolation::passMinE_(float energy,const HcalDetId id)const
{
  if(id.subdet()==HcalBarrel && energy>=eMinHB_) return true;
  else if(id.subdet()==HcalEndcap && energy>=eMinHE_) return true;
  else return false;
}

bool EgammaHLTHcalIsolation::passMinEt_(float et,const HcalDetId id)const
{
  if(id.subdet()==HcalBarrel && et>=etMinHB_) return true;
  else if(id.subdet()==HcalEndcap && et>=etMinHE_) return true;
  else return false;

}

bool EgammaHLTHcalIsolation::passDepth_(const HcalDetId id)const
{
  if(depth_==-1) return true; //I wish we had chosen 0 as all depths but EgammaTowerIsolation chose -1 and 0 as invalid
  else if(getEffectiveDepth(id)==depth_) return true;
  else return false;
    
}

//inspired from CaloTowersCreationAlgo::hcalChanStatusForCaloTower, we dont distingush from good from recovered and prob channels
bool EgammaHLTHcalIsolation::passCleaning_(const CaloRecHit* hit,const HcalSeverityLevelComputer* hcalSevLvlComp,
					   const HcalChannelQuality* hcalChanStatus)const
{
  if(hcalSevLvlComp==NULL || hcalChanStatus==NULL) return true; //return true if we dont have valid pointers

  const DetId id = hit->detid();
  
  const uint32_t recHitFlag = hit->flags();
  const uint32_t dbStatusFlag = hcalChanStatus->getValues(id)->getValue();
  
  int severityLevel = hcalSevLvlComp->getSeverityLevel(id,recHitFlag,dbStatusFlag);
  bool isRecovered = hcalSevLvlComp->recoveredRecHit(id,recHitFlag);
  
  if(severityLevel == 0) return true;
  else if(isRecovered) return useRecoveredHcalHits_;
  else if(severityLevel <= hcalAcceptSeverityLevel_) return true;
  else return false;
}



//this is the effective depth of the rec-hit, basically converts 3 depth towers to 2 depths and all barrel to depth 1
int EgammaHLTHcalIsolation::getEffectiveDepth(const HcalDetId id)
{
  int iEtaAbs = id.ietaAbs();
  int depth = id.depth();
  if(iEtaAbs<=17 ||
     (iEtaAbs<=29 && depth==1) ||
     (iEtaAbs>=27 && iEtaAbs<=29 && depth==2)){   
    return 1;
  }else return 2;
  
}
