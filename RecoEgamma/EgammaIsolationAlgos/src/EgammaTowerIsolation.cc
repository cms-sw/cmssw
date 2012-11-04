//*****************************************************************************
// File:      EgammaTowerIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"


EgammaTowerIsolation::EgammaTowerIsolation (float extRadius,
					    float intRadius,
					    float etLow,
					    signed int depth,
					    const CaloTowerCollection* towers ) : newAlgo(&extRadius,&intRadius,*towers), depth_(depth){}


double  EgammaTowerIsolation::getTowerEtSum (const reco::Candidate* cand, const std::vector<CaloTowerDetId> * detIdToExclude) const{
  EgammaTowerIsolationNew<1>::Sum sum;
  newAlgo.compute(sum,*cand, 
		  (detIdToExclude==0) ? nullptr : &((*detIdToExclude).front()),
		  (detIdToExclude==0) ? nullptr : (&(*detIdToExclude).back())+1
		  );
  
  switch(depth_){
  case AllDepths: return detIdToExclude==0 ? sum.he[0] : sum.heBC[0]; 
  case Depth1: return detIdToExclude==0 ? sum.he[0]-sum.h2[0] : sum.heBC[0]-sum.h2BC[0]; 
  case Depth2:return detIdToExclude==0 ? sum.h2[0] : sum.h2BC[0]; 
  default: return 0;
  }
  return 0;
}

