//*****************************************************************************
// File:      EgammaTowerIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include<cassert>


#ifdef ETISTATDEBUG
// #include<iostream>
namespace etiStat {
  Count::~Count() { 
    //    std::cout << "\nEgammaTowerIsolationNew " << create << "/" << comp << "/" << float(span)/float(comp)  
    //	      << std::endl<< std::endl;
    }

  Count Count::count;
}
#endif

namespace {
  struct TLS {   
    EgammaTowerIsolationNew<1> * newAlgo=nullptr;;
    const CaloTowerCollection* oldTowers=nullptr;;
    uint32_t id15=0;
  };
  thread_local static TLS tls;
}

EgammaTowerIsolation::EgammaTowerIsolation (float extRadiusI,
					    float intRadiusI,
					    float etLow,
					    signed int depth,
					    const CaloTowerCollection* towers ) :  
  depth_(depth), 
  extRadius(extRadiusI),
  intRadius(intRadiusI)
{
  assert(0==etLow);

  // extremely poor in quality  (test of performance)
  if (tls.newAlgo==nullptr ||  towers!=tls.oldTowers || towers->size()!=tls.newAlgo->nt || (towers->size()>15 && (*towers)[15].id()!=tls.id15)) {
    delete tls.newAlgo;
    tls.newAlgo = new EgammaTowerIsolationNew<1>(&extRadius,&intRadius,*towers);
    tls.oldTowers=towers;
    tls.id15 = towers->size()>15 ? (*towers)[15].id() : 0;
  }
}


double  EgammaTowerIsolation::getSum (bool et, reco::SuperCluster const & sc, const std::vector<CaloTowerDetId> * detIdToExclude) const{

  if (0!=detIdToExclude) assert(0==intRadius);

  // hack
  tls.newAlgo->setRadius(&extRadius,&intRadius);

  EgammaTowerIsolationNew<1>::Sum sum;
  tls.newAlgo->compute(et, sum, sc, 
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

