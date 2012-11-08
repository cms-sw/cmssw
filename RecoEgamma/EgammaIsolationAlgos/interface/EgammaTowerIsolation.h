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

#include<vector>


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

  EgammaTowerIsolationNew () : nt(0){}
  EgammaTowerIsolationNew (float extRadius[NC],
			   float intRadius[NC],
			   CaloTowerCollection const & towers) ;
 

  ~EgammaTowerIsolationNew() { delete[] mem;}
  
  void compute(bool et, Sum&sum, reco::Candidate const & cand,  CaloTowerDetId const * first,  CaloTowerDetId const * last) const {
    reco::SuperCluster const & sc =  *cand.get<reco::SuperClusterRef>().get();
    return compute(et,sum,sc,first,last);
  }
  void compute(bool et, Sum &sum, reco::SuperCluster const & sc,  CaloTowerDetId const * first,  CaloTowerDetId const * last) const;

  void setRadius(float extRadius[NC],float intRadius[NC]) {
    for (std::size_t i=0; i!=NCuts; ++i) {
    extRadius2_[i]=extRadius[i]*extRadius[i];
    intRadius2_[i]=intRadius[i]*intRadius[i];
  }


public:

  float extRadius2_[NCuts] ;
  float intRadius2_[NCuts] ;
  
  float maxEta;
  //SOA
  const uint32_t nt;
  float * eta;
  float * phi;
  float * he;
  float * h2;
  float * st;
  uint32_t * id;
  uint32_t * mem=nullptr;
  void initSoa() {
    mem = new uint32_t[nt*6];
    eta = (float*)(mem); phi = eta+nt; he = phi+nt; h2 = he+nt; st = h2+nt;
    id = (uint32_t*)(st) + nt;

 }
  
  
};

namespace etiStat {

  struct Count {
    uint32_t create=0;
    uint32_t comp=0;
    static Count count;
    ~Count();
  };

}


template<unsigned int NC>
inline
EgammaTowerIsolationNew<NC>::EgammaTowerIsolationNew(float extRadius[NC],
						     float intRadius[NC],
						     CaloTowerCollection const & towers) :
  maxEta(*std::max_element(extRadius,extRadius+NC)),
  nt(towers.size()) {
  if (nt==0) return;
  initSoa();

  etiStat::Count::count.create++;

  
  for (std::size_t i=0; i!=NCuts; ++i) {
    extRadius2_[i]=extRadius[i]*extRadius[i];
    intRadius2_[i]=intRadius[i]*intRadius[i];
  }
  
  // sort in eta  (kd-tree anoverkill,does not vectorize...)
  
  
  uint32_t index[nt];
  float e[nt];
  for (std::size_t i=0; i!=nt; ++i) {
    e[i]=towers[i].eta();
    index[i]=i;
    std::push_heap(index,index+i+1,[&e](uint32_t i, uint32_t j){ return e[i]<e[j];});
  }
  std::sort_heap(index,index+nt,[&e](uint32_t i, uint32_t j){ return e[i]<e[j];});
  
  
  for (std::size_t i=0;i!=nt; ++i) {
    auto j = index[i];
    eta[i]=towers[j].eta();
    phi[i]=towers[j].phi();
    id[i]=towers[j].id();
    st[i] =  1.f/std::cosh(eta[i]); // std::sin(towers[j].theta());   //;
    he[i] = towers[j].hadEnergy();
    h2[i] = towers[j].hadEnergyHeOuterLayer();
  }
}


template<unsigned int NC>
inline
void
EgammaTowerIsolationNew<NC>::compute(bool et, Sum &sum, reco::SuperCluster const & sc,  CaloTowerDetId const * first,  CaloTowerDetId const * last) const {
  if (nt==0) return;

  etiStat::Count::count.comp++;

  float candEta = sc.eta();
  float candPhi = sc.phi();

  
  auto lb = std::lower_bound(eta,eta+nt,candEta-maxEta);
  auto ub = std::upper_bound(lb,eta+nt,candEta+maxEta);
  uint32_t il = lb-eta;
  uint32_t iu = std::min(nt,uint32_t(ub-eta+1));
  
  bool ok[iu-il];
  for (std::size_t i=il;i!=iu; ++i)
    ok[i-il] = (std::find(first,last,id[i])==last);
  


  // should be restricted in eta....
  for (std::size_t i=il;i!=iu; ++i) {
    float dr2 = reco::deltaR2(candEta,candPhi,eta[i], phi[i]);
    float tt = et ? st[i] : 1.f;
    for (std::size_t j=0; j!=NCuts; ++j) {
      if (dr2<extRadius2_[j]) {
	if (dr2>=intRadius2_[j]) {
	  sum.he[j] +=he[i]*tt;
	  sum.h2[j] +=h2[i]*tt;
	}
	if(ok[i-il]) {
	  sum.heBC[j] +=he[i]*tt;
	  sum.h2BC[j] +=h2[i]*tt;
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
  
  double getTowerEtSum (const reco::Candidate* cand, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const{
    reco::SuperCluster const & sc =  *cand->get<reco::SuperClusterRef>().get();
    return getSum(true,sc,detIdToExclude);
  }
  double  getTowerESum (const reco::Candidate* cand, const std::vector<CaloTowerDetId> * detIdToExclude=0) const{
    reco::SuperCluster const & sc =  *cand->get<reco::SuperClusterRef>().get();
    return getSum(false,sc,detIdToExclude);
  }
  double getTowerEtSum (reco::SuperCluster const * sc, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const{
    return getSum(true,*sc,detIdToExclude);
  }
  double  getTowerESum (reco::SuperCluster const * sc, const std::vector<CaloTowerDetId> * detIdToExclude=0) const{
    return getSum(false,*sc,detIdToExclude);
  }
  private:
  double getSum (bool et, reco::SuperCluster const & sc, const std::vector<CaloTowerDetId> * detIdToExclude) const;

  
private:
  static EgammaTowerIsolationNew<1> * newAlgo;
  static const CaloTowerCollection* oldTowers;
  static uint32_t id15;
  signed int depth_;
  bool rzero;
};



#endif
