#ifndef EgammaTowerIsolation_h
#define EgammaTowerIsolation_h

//*****************************************************************************
// File:      EgammaTowerIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//  Adding feature to exclude towers used by H/E
//
//  11/11/12 Hack by VI to make it 100 times faster
//=============================================================================
//*****************************************************************************

#include <vector>

//CMSSW includes
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <atomic>

#include "DataFormats/Math/interface/deltaR.h"

/*
  for each set of cuts it will compute Et for all, depth1 and depth2 twice:
  one between inner and outer and once inside outer vetoid the tower to excude

 */
template <unsigned int NC>
class EgammaTowerIsolationNew {
public:
  struct Sum {
    Sum() : he{0}, h2{0}, heBC{0}, h2BC{0} {}
    float he[NC];
    float h2[NC];
    float heBC[NC];
    float h2BC[NC];
  };

  // number of cuts
  constexpr static unsigned int NCuts = NC;

  //constructors

  EgammaTowerIsolationNew() : nt(0) {}
  EgammaTowerIsolationNew(float extRadius[NC], float intRadius[NC], CaloTowerCollection const& towers);

  ~EgammaTowerIsolationNew() { delete[] mem; }

  template <typename I>
  void compute(bool et, Sum& sum, reco::Candidate const& cand, I first, I last) const {
    reco::SuperCluster const* sc = cand.get<reco::SuperClusterRef>().get();
    if (sc) {
      compute(et, sum, *sc, first, last);
    }
  }
  template <typename I>
  void compute(bool et, Sum& sum, reco::SuperCluster const& sc, I first, I last) const;

  void setRadius(float const extRadius[NC], float const intRadius[NC]) {
    for (std::size_t i = 0; i != NCuts; ++i) {
      extRadius2_[i] = extRadius[i] * extRadius[i];
      intRadius2_[i] = intRadius[i] * intRadius[i];
    }
    maxEta = *std::max_element(extRadius, extRadius + NC);
  }

public:
  float extRadius2_[NCuts];
  float intRadius2_[NCuts];

  float maxEta;
  //SOA
  const uint32_t nt;
  float* eta;
  float* phi;
  float* he;
  float* h2;
  float* st;
  uint32_t* id;
  uint32_t* mem = nullptr;
  void initSoa() {
    mem = new uint32_t[nt * 6];
    eta = (float*)(mem);
    phi = eta + nt;
    he = phi + nt;
    h2 = he + nt;
    st = h2 + nt;
    id = (uint32_t*)(st) + nt;
  }
};

/*#define ETISTATDEBUG*/
#ifdef ETISTATDEBUG
namespace etiStat {

  struct Count {
    std::atomic<uint32_t> create = 0;
    std::atomic<uint32_t> comp = 0;
    std::atomic<uint32_t> span = 0;
    static Count count;
    ~Count();
  };

}  // namespace etiStat
#endif

template <unsigned int NC>
inline EgammaTowerIsolationNew<NC>::EgammaTowerIsolationNew(float extRadius[NC],
                                                            float intRadius[NC],
                                                            CaloTowerCollection const& towers)
    : maxEta(*std::max_element(extRadius, extRadius + NC)), nt(towers.size()) {
  if (nt == 0)
    return;
  initSoa();

#ifdef ETISTATDEBUG
  etiStat::Count::count.create++;
#endif

  for (std::size_t i = 0; i != NCuts; ++i) {
    extRadius2_[i] = extRadius[i] * extRadius[i];
    intRadius2_[i] = intRadius[i] * intRadius[i];
  }

  // sort in eta  (kd-tree anoverkill,does not vectorize...)
  uint32_t index[nt];
#ifdef __clang__
  std::vector<float> e(nt);
#else
  float e[nt];
#endif
  for (std::size_t k = 0; k != nt; ++k) {
    e[k] = towers[k].eta();
    index[k] = k;
    std::push_heap(index, index + k + 1, [&e](uint32_t i, uint32_t j) { return e[i] < e[j]; });
  }
  std::sort_heap(index, index + nt, [&e](uint32_t i, uint32_t j) { return e[i] < e[j]; });

  for (std::size_t i = 0; i != nt; ++i) {
    auto j = index[i];
    eta[i] = towers[j].eta();
    phi[i] = towers[j].phi();
    id[i] = towers[j].id();
    st[i] = 1.f / std::cosh(eta[i]);  // std::sin(towers[j].theta());   //;
    he[i] = towers[j].hadEnergy();
    h2[i] = towers[j].hadEnergyHeOuterLayer();
  }
}

template <unsigned int NC>
template <typename I>
inline void EgammaTowerIsolationNew<NC>::compute(
    bool et, Sum& sum, reco::SuperCluster const& sc, I first, I last) const {
  if (nt == 0)
    return;

#ifdef ETISTATDEBUG
  etiStat::Count::count.comp++;
#endif

  float candEta = sc.eta();
  float candPhi = sc.phi();

  auto lb = std::lower_bound(eta, eta + nt, candEta - maxEta);
  auto ub = std::upper_bound(lb, eta + nt, candEta + maxEta);
  uint32_t il = lb - eta;
  uint32_t iu = std::min(nt, uint32_t(ub - eta + 1));

#ifdef ETISTATDEBUG
  etiStat::Count::count.span += (iu - il);
#endif

  // should be restricted in eta....
  for (std::size_t i = il; i != iu; ++i) {
    float dr2 = reco::deltaR2(candEta, candPhi, eta[i], phi[i]);
    float tt = et ? st[i] : 1.f;
    for (std::size_t j = 0; j != NCuts; ++j) {
      if (dr2 < extRadius2_[j]) {
        if (dr2 >= intRadius2_[j]) {
          sum.he[j] += he[i] * tt;
          sum.h2[j] += h2[i] * tt;
        }
        if (std::find(first, last, id[i]) == last) {
          sum.heBC[j] += he[i] * tt;
          sum.h2BC[j] += h2[i] * tt;
        }
      }
    }
  }
}

class EgammaTowerIsolation {
public:
  enum HcalDepth { AllDepths = -1, Undefined = 0, Depth1 = 1, Depth2 = 2 };

  //constructors
  EgammaTowerIsolation(
      float extRadiusI, float intRadiusI, float etLow, signed int depth, const CaloTowerCollection* towers);

  double getTowerEtSum(const reco::Candidate* cand, const std::vector<CaloTowerDetId>* detIdToExclude = nullptr) const {
    reco::SuperCluster const& sc = *cand->get<reco::SuperClusterRef>().get();
    return getSum(true, sc, detIdToExclude);
  }
  double getTowerESum(const reco::Candidate* cand, const std::vector<CaloTowerDetId>* detIdToExclude = nullptr) const {
    reco::SuperCluster const& sc = *cand->get<reco::SuperClusterRef>().get();
    return getSum(false, sc, detIdToExclude);
  }
  double getTowerEtSum(reco::SuperCluster const* sc,
                       const std::vector<CaloTowerDetId>* detIdToExclude = nullptr) const {
    return getSum(true, *sc, detIdToExclude);
  }
  double getTowerESum(reco::SuperCluster const* sc, const std::vector<CaloTowerDetId>* detIdToExclude = nullptr) const {
    return getSum(false, *sc, detIdToExclude);
  }

private:
  double getSum(bool et, reco::SuperCluster const& sc, const std::vector<CaloTowerDetId>* detIdToExclude) const;

private:
  signed int depth_;
  float extRadius;
  float intRadius;
};

#endif
