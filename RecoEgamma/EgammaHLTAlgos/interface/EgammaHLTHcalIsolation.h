#ifndef EgammaHLTAlgos_EgammaHLTHcalIsolation_h
#define EgammaHLTAlgos_EgammaHLTHcalIsolation_h
// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTHcalIsolation
// 
/**\class EgammaHLTHcalIsolation EgammaHLTHcalIsolation.h RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h

 Description: sum pt hcal hits in cone around egamma candidate

 Usage:
    <usage>

*/
//
// Original Author:  Monica Vazquez Acosta - CERN
//         Created:  Tue Jun 13 12:18:35 CEST 2006
// $Id: EgammaHLTHcalIsolation.h,v 1.4 2010/08/12 15:25:02 sharper Exp $
// modifed by Sam Harper (RAL) 27/7/10

//the class aims to as closely as possible emulate the RECO HCAL isolation
//which uses CaloTowers
//now CaloTowers are just rec-hits with E>Ethres except for tower 28 depth 3
//which equally splits the energy between tower 28 and 29

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

class HcalSeverityLevelComputer;
class HcalChannelQuality;

class EgammaHLTHcalIsolation
{

 public:
  
  EgammaHLTHcalIsolation(float eMinHB,float eMinHE,float etMinHB,float etMinHE,
			 float innerCone,float outerCone,int depth) :
    eMinHB_(eMinHB),eMinHE_(eMinHE),etMinHB_(etMinHB),etMinHE_(etMinHE),
    innerCone_(innerCone),outerCone_(outerCone),depth_(depth),
    useRecoveredHcalHits_(1),hcalAcceptSeverityLevel_(9){}//{std::cout <<"innerCone "<<innerCone<<" outerCone "<<outerCone<<std::endl;}
    
    //first is the sum E, second is the sum Et
    std::pair<float,float> getSum(float candEta,float candPhi,
				  const HBHERecHitCollection* hbhe, const CaloGeometry* geometry,
				  const HcalSeverityLevelComputer* hcalSevLvlAlgo=NULL,
				  const HcalChannelQuality* dbHcalChStatus=NULL)const;
    float getESum(float candEta,float candPhi, 
		  const HBHERecHitCollection* hbhe,
		  const CaloGeometry* geometry)const{return getSum(candEta,candPhi,hbhe,geometry).first;}
    float getEtSum(float candEta,float candPhi, 
		   const HBHERecHitCollection* hbhe, 
		   const CaloGeometry* geometry)const{return getSum(candEta,candPhi,hbhe,geometry).second;}
    float getESum(float candEta,float candPhi, 
		  const HBHERecHitCollection* hbhe,
		  const CaloGeometry* geometry,
		  const HcalSeverityLevelComputer* hcalSevLvlAlgo,
		  const HcalChannelQuality* dbHcalChStatus)const{return getSum(candEta,candPhi,hbhe,geometry,
									       hcalSevLvlAlgo,dbHcalChStatus).first;}
    float getEtSum(float candEta,float candPhi, 
		   const HBHERecHitCollection* hbhe, 
		   const CaloGeometry* geometry,
		   const HcalSeverityLevelComputer* hcalSevLvlAlgo,
		   const HcalChannelQuality* dbHcalChStatus)const{return getSum(candEta,candPhi,hbhe,geometry,
										hcalSevLvlAlgo,dbHcalChStatus).second;}

    //this is the effective depth of the rec-hit, basically converts 3 depth towers to 2 depths and all barrel to depth 1
    //this is defined when making the calotowers
    static int getEffectiveDepth(const HcalDetId id);

 private:  
    bool acceptHit_(const HcalDetId id,const GlobalPoint& pos,const float hitEnergy,
		    const float candEta,const float candPhi)const; 
    bool passMinE_(float energy,const HcalDetId id)const;
    bool passMinEt_(float et,const HcalDetId id)const;
    bool passDepth_(const HcalDetId id)const;
    //inspired from CaloTowersCreationAlgo::hcalChanStatusForCaloTower, we dont distingush from good from recovered and prob channels
    bool passCleaning_(const CaloRecHit* hit,const HcalSeverityLevelComputer* hcalSevLvlComp,
		       const HcalChannelQuality* hcalChanStatus)const;

 private:

  // ---------- member data --------------------------------
   // Parameters of isolation cone geometry. 
  float eMinHB_;
  float eMinHE_;
  float etMinHB_;
  float etMinHE_;
  float innerCone_;
  float outerCone_;
  int depth_;
  
  //cleaning parameters
  bool useRecoveredHcalHits_;
  int hcalAcceptSeverityLevel_;
};


#endif
