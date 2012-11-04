#ifndef EgammaTowerIsolation_h
#define EgammaTowerIsolation_h

//*****************************************************************************
// File:      EgammaTowerIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//  Adding feature to exclude towers used by H/E
//=============================================================================
//*****************************************************************************

//CMSSW includes
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"


#include <cmath>
#include <algorithm>
#include <cstdint>

#include "DataFormats/Math/interface/deltaR.h"


/*
  for each set of cuts it will compute Et for all, depth1 and depth2 twice:
  one between inner and outer and once inside outer vetoid the tower to excude

 */
template<unsigned int NC>
class EgammaTowerIsolationNew {
 public:

  struct Sum {
    Sum(): he{0},h2{0},heBC{0},h2BC{0}
    {}
    float he[NC];
    float h2[NC];
    float heBC[NC];
    float h2BC[NC];
  };

  // number of cuts
  constexpr static unsigned int NCuts = NC;

  //constructors
  EgammaTowerIsolationNew (float extRadius[NC],
			   float intRadius[NC],
			   CaloTowerCollection const & towers) ;
 

  ~EgammaTowerIsolationNew() { delete[] mem;}
  
  void compute(Sum&sum, reco::Candidate const & cand,  CaloTowerDetId const * first,  CaloTowerDetId const * last) const;
  
private:

  float extRadius2_[NCuts] ;
  float intRadius2_[NCuts] ;
  
  //SOA
  const uint32_t nt;
  float * eta;
  float * phi;
  float * he;
  float * h2;
  uint32_t * id;
  uint32_t * mem=nullptr;
  void init() {
    mem = new uint32_t[nt*5];
    eta = (float*)(mem); phi = eta+nt; he = phi+nt; h2 = he+nt; id = (uint32_t*)(h2) + nt;
  }
  
  
};




template<unsigned int NC>
inline
EgammaTowerIsolationNew<NC>::EgammaTowerIsolationNew(float extRadius[NC],
						     float intRadius[NC],
						     CaloTowerCollection const & towers) : nt(towers.size()) {
  if (nt==0) return;
  init();
  
  for (unsigned int i=0; i!=NCuts; ++i) {
    extRadius2_[i]=extRadius[i]*extRadius[i];
    intRadius2_[i]=intRadius[i]*intRadius[i];
  }
  
  // sort in eta  (kd-tree anoverkill,does not vectorize...)
  
  
  uint32_t index[nt];
  float tmp[nt]; float * p=tmp;
  for (uint32_t i=0; i!=nt; ++i) {
    tmp[i]=towers[i].eta();
    index[i]=i;
  }
  std::sort(index,index+nt,[p](uint32_t i, uint32_t j){ return p[i]<p[j];});
  
  
  for ( uint32_t i=0;i!=nt; ++i) {
    auto j = index[i];
    eta[i]=towers[j].eta();
    phi[i]=towers[j].phi();
    id[i]=towers[i].id();
    float st = std::cosh(eta[i]);
    he[i] = st*towers[j].hadEnergy();
    h2[i] = st*towers[j].hadEnergyHeOuterLayer();
  }
  
  
}

template<unsigned int NC>
inline
void
EgammaTowerIsolationNew<NC>::compute(Sum &sum, reco::Candidate const & cand,  CaloTowerDetId const * first,  CaloTowerDetId const * last) const {
  if (nt==0) return;
  
  reco::SuperCluster const & sc =  *cand.get<reco::SuperClusterRef>().get();
  float candEta = sc.eta();
  float candPhi = sc.phi();
  
  bool ok[nt];
  for ( uint32_t i=0;i!=nt; ++i)
    ok[i] = (std::find(first,last,id[i])==last);
  
  // should be restricted in eta....
  for (uint32_t i=0;i!=nt; ++i) {
    float dr2 = reco::deltaR2(candEta,candPhi,eta[i], phi[i]);
    for (unsigned int j=0; j!=NCuts; ++j) {
      if (dr2<extRadius2_[j]) {
	if (dr2>=intRadius2_[j]) {
	  sum.he[j] +=he[i];
	  sum.h2[j] +=h2[i];
	}
	if(ok[i]) {
	  sum.heBC[j] +=he[i];
	  sum.h2BC[j] +=h2[i];
	}
      }
    }
  }
}

class EgammaTowerIsolation {
public:
  
  enum HcalDepth{AllDepths=-1,Undefined=0,Depth1=1,Depth2=2};
  
  //constructors
  EgammaTowerIsolation (float extRadius,
			float intRadius,
			float etLow,
			signed int depth,
			const CaloTowerCollection* towers );
  
  double getTowerEtSum (const reco::Candidate* cand, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const;
  
private:
  EgammaTowerIsolationNew<1> newAlgo;
  signed int depth_;
};



#endif
