//*****************************************************************************
// File:      EgammaTowerIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include <cassert>
#include <memory>

#ifdef ETISTATDEBUG
// #include<iostream>
namespace etiStat {
  Count::~Count() {
    //    std::cout << "\nEgammaTowerIsolationNew " << create << "/" << comp << "/" << float(span)/float(comp)
    //	      << std::endl<< std::endl;
  }

  Count Count::count;
}  // namespace etiStat
#endif

namespace {
  struct TLS {
    std::unique_ptr<EgammaTowerIsolationNew<1>> newAlgo = nullptr;
    ;
    const CaloTowerCollection* oldTowers = nullptr;
    ;
    uint32_t id15 = 0;
  };
  thread_local TLS tls;
}  // namespace

EgammaTowerIsolation::EgammaTowerIsolation(
    float extRadiusI, float intRadiusI, float etLow, signed int depth, const CaloTowerCollection* towers)
    : depth_(depth), extRadius(extRadiusI), intRadius(intRadiusI) {
  assert(0 == etLow);

  // extremely poor in quality  (test of performance)
  if (tls.newAlgo.get() == nullptr || towers != tls.oldTowers || towers->size() != tls.newAlgo->nt ||
      (towers->size() > 15 && (*towers)[15].id() != tls.id15)) {
    tls.newAlgo = std::make_unique<EgammaTowerIsolationNew<1>>(&extRadius, &intRadius, *towers);
    tls.oldTowers = towers;
    tls.id15 = towers->size() > 15 ? (*towers)[15].id() : 0;
  }
}

double EgammaTowerIsolation::getSum(bool et,
                                    reco::SuperCluster const& sc,
                                    const std::vector<CaloTowerDetId>* detIdToExclude) const {
  if (nullptr != detIdToExclude)
    assert(0 == intRadius);

  // hack
  tls.newAlgo->setRadius(&extRadius, &intRadius);

  EgammaTowerIsolationNew<1>::Sum sum;
  if (detIdToExclude == nullptr) {
    tls.newAlgo->compute(
        et, sum, sc, static_cast<CaloTowerDetId const*>(nullptr), static_cast<CaloTowerDetId const*>(nullptr));
  } else {
    tls.newAlgo->compute(et, sum, sc, detIdToExclude->cbegin(), detIdToExclude->cend());
  }

  switch (depth_) {
    case AllDepths:
      return detIdToExclude == nullptr ? sum.he[0] : sum.heBC[0];
    case Depth1:
      return detIdToExclude == nullptr ? sum.he[0] - sum.h2[0] : sum.heBC[0] - sum.h2BC[0];
    case Depth2:
      return detIdToExclude == nullptr ? sum.h2[0] : sum.h2BC[0];
    default:
      return 0;
  }
  return 0;
}
