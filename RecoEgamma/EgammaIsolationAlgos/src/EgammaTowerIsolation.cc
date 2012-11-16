//*****************************************************************************
// File:      EgammaTowerIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>
#include <math.h>

//ROOT includes
#include <Math/VectorUtil.h>


//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace ROOT::Math::VectorUtil ;

EgammaTowerIsolation::EgammaTowerIsolation (double extRadius,
				      double intRadius,
				      double etLow,
				      signed int depth,
				      const CaloTowerCollection* towercollection)   :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  depth_(depth),
  towercollection_(towercollection)
{
}

EgammaTowerIsolation::~EgammaTowerIsolation ()
{
}




double EgammaTowerIsolation::getTowerEtSum(const reco::Candidate* photon, const std::vector<CaloTowerDetId> * detIdToExclude ) const
{
  return getTowerEtSum( photon->get<reco::SuperClusterRef>().get(), detIdToExclude );
}

double EgammaTowerIsolation::getTowerEtSum(const reco::SuperCluster* sc, const std::vector<CaloTowerDetId> * detIdToExclude ) const
{
  double candEta=sc->eta();
  double candPhi=sc->phi();
  double ptSum =0.;

  //loop over towers
  for(CaloTowerCollection::const_iterator trItr = towercollection_->begin(); trItr != towercollection_->end(); ++trItr){

    // skip the towers to exclude
    if ( detIdToExclude )
     {
      std::vector<CaloTowerDetId>::const_iterator itcheck=find(detIdToExclude->begin(),detIdToExclude->end(),trItr->id());
      if (itcheck != detIdToExclude->end())
	      continue;
     }

    double this_pt=0;
    switch(depth_){
    case AllDepths: this_pt = trItr->hadEt();break;
    case Depth1: this_pt = trItr->ietaAbs()<18 || trItr->ietaAbs()>29 ? trItr->hadEt() : trItr->hadEnergyHeInnerLayer()*sin(trItr->p4().theta());break;
    case Depth2: this_pt = trItr->hadEnergyHeOuterLayer()*sin(trItr->p4().theta());break;
    default: throw cms::Exception("Configuration Error") << "EgammaTowerIsolation: Depth " << depth_ << " not known. "; break;
    }

    if ( this_pt < etLow_ )
      continue ;

    double towerEta=trItr->eta();
    double towerPhi=trItr->phi();
    double twoPi= 2*M_PI;
    if(towerPhi<0) towerPhi+=twoPi;
    if(candPhi<0) candPhi+=twoPi;
    double deltaPhi=fabs(towerPhi-candPhi);
    if(deltaPhi>twoPi) deltaPhi-=twoPi;
    if(deltaPhi>M_PI) deltaPhi=twoPi-deltaPhi;
    double deltaEta = towerEta - candEta;

    double dr = deltaEta*deltaEta + deltaPhi*deltaPhi;
    if( dr < extRadius_*extRadius_ &&
        dr >= intRadius_*intRadius_ )
     { ptSum += this_pt ; }

   }//end loop over tracks

  return ptSum ;

 }


double EgammaTowerIsolation::getTowerESum(const reco::Candidate* photon, const std::vector<CaloTowerDetId> * detIdToExclude ) const
{
  return getTowerESum( photon->get<reco::SuperClusterRef>().get(), detIdToExclude );
}

double EgammaTowerIsolation::getTowerESum(const reco::SuperCluster* sc, const std::vector<CaloTowerDetId> * detIdToExclude ) const
{
  double candEta=sc->eta();
  double candPhi=sc->phi();
  double eSum =0.;

  //loop over towers
  for(CaloTowerCollection::const_iterator trItr = towercollection_->begin(); trItr != towercollection_->end(); ++trItr){

    // skip the towers to exclude
    if( detIdToExclude ) {
      std::vector<CaloTowerDetId>::const_iterator itcheck=find(detIdToExclude->begin(),detIdToExclude->end(),trItr->id());
      if (itcheck != detIdToExclude->end())
	continue;
    }

    double this_e=0;
    switch(depth_){
    case AllDepths: this_e = trItr->hadEnergy();break;
    case Depth1: this_e = trItr->ietaAbs()<18 || trItr->ietaAbs()>29 ? trItr->hadEnergy() : trItr->hadEnergyHeInnerLayer();break;
    case Depth2: this_e = trItr->hadEnergyHeOuterLayer();break;
    default: throw cms::Exception("Configuration Error") << "EgammaTowerIsolation: Depth " << depth_ << " not known. "; break;
    }

    if ( this_e*sin(trItr->p4().theta()) < etLow_ )
      continue ;

    double towerEta=trItr->eta();
    double towerPhi=trItr->phi();
    double twoPi= 2*M_PI;
    if(towerPhi<0) towerPhi+=twoPi;
    if(candPhi<0) candPhi+=twoPi;
    double deltaPhi=fabs(towerPhi-candPhi);
    if(deltaPhi>twoPi) deltaPhi-=twoPi;
    if(deltaPhi>M_PI) deltaPhi=twoPi-deltaPhi;
    double deltaEta = towerEta - candEta;


    double dr = deltaEta*deltaEta + deltaPhi*deltaPhi;
    if( dr < extRadius_*extRadius_ &&
        dr >= intRadius_*intRadius_ )
      {
	eSum += this_e;
      }

  }//end loop over tracks

  return eSum;
}

