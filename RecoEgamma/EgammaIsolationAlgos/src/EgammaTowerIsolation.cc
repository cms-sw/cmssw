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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

using namespace ROOT::Math::VectorUtil ;

EgammaTowerIsolation::EgammaTowerIsolation (double extRadius,
				      double intRadius,
				      double etLow,
				      const CaloTowerCollection* towercollection)   :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  towercollection_(towercollection)  
{
}

EgammaTowerIsolation::~EgammaTowerIsolation ()
{
}



// unified acces to isolations
double EgammaTowerIsolation::getTowerEtSum(const reco::Candidate* photon) const  
{
  double ptSum =0.;


  //Take the SC position
  reco::SuperClusterRef sc = photon->get<reco::SuperClusterRef>();
  math::XYZPoint theCaloPosition = sc.get()->position();
  double candEta=sc.get()->eta();
  double candPhi=sc.get()->phi();

  //loop over tracks
  for(CaloTowerCollection::const_iterator trItr = towercollection_->begin(); trItr != towercollection_->end(); ++trItr){
    
    double this_pt  = trItr->hadEt();
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
      {
	ptSum += this_pt;
      }
    
  }//end loop over tracks

  return ptSum;
}

