//*****************************************************************************
// File:      EgammaTowerIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <algorithm>
#include <cmath>


//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaR.h"

EgammaTowerIsolation::EgammaTowerIsolation (double extRadius,
				      double intRadius,
				      double etLow,
				      signed int depth,
				      const CaloTowerCollection* towercollection)   :
  extRadius2_(extRadius*extRadius),
  intRadius2_(intRadius*intRadius),
  etLow_(etLow),
  depth_(depth),
  towercollection_(towercollection)
{
    switch(depth_){
    case AllDepths: break;
    case Depth1:;break;
    case Depth2: break;
    default: throw cms::Exception("Configuration Error") << "EgammaTowerIsolation: Depth " << depth_ << " not known. "; break;
    }
}

EgammaTowerIsolation::~EgammaTowerIsolation ()
{
}


double EgammaTowerIsolation::getTowerEtSum(const reco::Candidate* photon, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  return getTowerEtSum(photon->eta(),photon->phi(), detIdToExclude );
}

double EgammaTowerIsolation::getTowerEtSum(const reco::SuperCluster* sc, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  return getTowerEtSum(sc->eta(),sc->phi(), detIdToExclude );
}


double EgammaTowerIsolation::getTowerEtSum(float candEta, float candPhi, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  
  double ptSum =0.;
  
  //loop over towers
  for(CaloTowerCollection::const_iterator trItr = towercollection_->begin(); trItr != towercollection_->end(); ++trItr){
    
    // skip the towers to exclude
    if ( detIdToExclude )
     {
       std::vector<CaloTowerDetId>::const_iterator itcheck=std::find(detIdToExclude->begin(),detIdToExclude->end(),trItr->id());
      if (itcheck != detIdToExclude->end())
	      continue;
     }

    double this_pt=0;
    switch(depth_){
    case AllDepths: this_pt = trItr->hadEt();break;
    case Depth1: this_pt = (trItr->ietaAbs()<18 || trItr->ietaAbs()>29) ? trItr->hadEt() : trItr->hadEnergyHeInnerLayer()*std::sin(trItr->theta());break;
    case Depth2: this_pt = trItr->hadEnergyHeOuterLayer()*std::sin(trItr->theta());break;
    default:  break;
    }

    if ( this_pt < etLow_ )
      continue ;

    float towerEta=trItr->eta();
    float towerPhi=trItr->phi();
    

    float dr2 = reco::deltaR2(candEta,candPhi,towerEta, towerPhi);
    if( dr2 < extRadius2_ &&
        dr2 >= intRadius2_ )
     { ptSum += this_pt ; }

   }//end loop over tracks

  return ptSum ;

 }


double EgammaTowerIsolation::getTowerESum(const reco::Candidate* photon, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  return getTowerESum(photon->eta(),photon->phi(), detIdToExclude );
}

double EgammaTowerIsolation::getTowerESum(const reco::SuperCluster* sc, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  return getTowerESum(sc->eta(),sc->phi(), detIdToExclude );
}


double EgammaTowerIsolation::getTowerESum(float candEta, float candPhi, const std::vector<CaloTowerDetId> * detIdToExclude ) const {
  
  double eSum =0.;
  
  //loop over towers
  for(CaloTowerCollection::const_iterator trItr = towercollection_->begin(); trItr != towercollection_->end(); ++trItr){
    
    // skip the towers to exclude
    if( detIdToExclude ) {
      std::vector<CaloTowerDetId>::const_iterator itcheck=std::find(detIdToExclude->begin(),detIdToExclude->end(),trItr->id());
      if (itcheck != detIdToExclude->end())
	continue;
    }

    double this_e=0;
    switch(depth_){
    case AllDepths: this_e = trItr->hadEnergy();break;
    case Depth1: this_e = (trItr->ietaAbs()<18 || trItr->ietaAbs()>29) ? trItr->hadEnergy() : trItr->hadEnergyHeInnerLayer();break;
    case Depth2: this_e = trItr->hadEnergyHeOuterLayer();break;
    default: break;
    }

    if ( this_e*std::sin(trItr->theta()) < etLow_ )
      continue ;

    float towerEta=trItr->eta();
    float towerPhi=trItr->phi();
   
    float dr2 = reco::deltaR2(candEta,candPhi,towerEta, towerPhi);

    if( dr2 < extRadius2_ &&
        dr2 >= intRadius2_ )
      {
	eSum += this_e;
      }

  }//end loop over tracks

  return eSum;
}

