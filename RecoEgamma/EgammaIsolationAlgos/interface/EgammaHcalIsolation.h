#ifndef EgammaIsolationAlgos_EgammaHcalIsolation_h
#define EgammaIsolationAlgos_EgammaHcalIsolation_h
//*****************************************************************************
// File:      EgammaHcalIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <array>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"

// sum helper functions
double scaleToE(const double &eta);
double scaleToEt(const double &eta);

class EgammaHcalIsolation {
public:
  enum class InclusionRule : int { withinConeAroundCluster = 0, isBehindClusterSeed = 1 };
  using arrayHB = std::array<double, 4>;
  using arrayHE = std::array<double, 7>;

  // constructors
  EgammaHcalIsolation(InclusionRule extIncRule,
                      double extRadius,
                      InclusionRule intIncRule,
                      double intRadius,
                      const arrayHB &eThresHB,
                      const arrayHB &etThresHB,
                      int maxSeverityHB,
                      const arrayHE &eThresHE,
                      const arrayHE &etThresHE,
                      int maxSeverityHE,
                      const HBHERecHitCollection &mhbhe,
                      edm::ESHandle<CaloGeometry> caloGeometry,
                      edm::ESHandle<HcalTopology> hcalTopology,
                      edm::ESHandle<HcalChannelQuality> hcalChStatus,
                      edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputer,
                      edm::ESHandle<CaloTowerConstituentsMap> towerMap);

  EgammaHcalIsolation(InclusionRule extIncRule,
                      double extRadius,
                      InclusionRule intIncRule,
                      double intRadius,
                      const arrayHB &eThresHB,
                      const arrayHB &etThresHB,
                      int maxSeverityHB,
                      const arrayHE &eThresHE,
                      const arrayHE &etThresHE,
                      int maxSeverityHE,
                      const HBHERecHitCollection &mhbhe,
                      const CaloGeometry &caloGeometry,
                      const HcalTopology &hcalTopology,
                      const HcalChannelQuality &hcalChStatus,
                      const HcalSeverityLevelComputer &hcalSevLvlComputer,
                      const CaloTowerConstituentsMap &towerMap);

  double getHcalESum(const reco::Candidate *c, int depth) const {
    return getHcalESum(c->get<reco::SuperClusterRef>().get(), depth);
  }
  double getHcalEtSum(const reco::Candidate *c, int depth) const {
    return getHcalEtSum(c->get<reco::SuperClusterRef>().get(), depth);
  }
  double getHcalESum(const reco::SuperCluster *sc, int depth) const { return getHcalESum(sc->position(), depth); }
  double getHcalEtSum(const reco::SuperCluster *sc, int depth) const { return getHcalEtSum(sc->position(), depth); }
  double getHcalESum(const math::XYZPoint &p, int depth) const {
    return getHcalESum(GlobalPoint(p.x(), p.y(), p.z()), depth);
  }
  double getHcalEtSum(const math::XYZPoint &p, int depth) const {
    return getHcalEtSum(GlobalPoint(p.x(), p.y(), p.z()), depth);
  }
  double getHcalESum(const GlobalPoint &pclu, int depth) const { return getHcalSum(pclu, depth, 0, 0, 0, &scaleToE); }
  double getHcalEtSum(const GlobalPoint &pclu, int depth) const { return getHcalSum(pclu, depth, 0, 0, 0, &scaleToEt); }

  double getHcalESumBc(const reco::Candidate *c, int depth) const {
    return getHcalESumBc(c->get<reco::SuperClusterRef>().get(), depth);
  }
  double getHcalEtSumBc(const reco::Candidate *c, int depth) const {
    return getHcalEtSumBc(c->get<reco::SuperClusterRef>().get(), depth);
  }
  double getHcalESumBc(const reco::SuperCluster *sc, int depth) const {
    const auto tower = egamma::towerOf(*(sc->seed()), towerMap_);

    if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::withinConeAroundCluster)
      return getHcalESumBc(sc->position(), depth, tower.ieta(), tower.iphi(), -1);
    else if (extIncRule_ == InclusionRule::withinConeAroundCluster and
             intIncRule_ == InclusionRule::isBehindClusterSeed)
      return getHcalESumBc(sc->position(), depth, tower.ieta(), tower.iphi(), 1);

    return getHcalESumBc(sc->position(), depth, tower.ieta(), tower.iphi(), 0);
  }
  double getHcalEtSumBc(const reco::SuperCluster *sc, int depth) const {
    const auto tower = egamma::towerOf(*(sc->seed()), towerMap_);

    if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::withinConeAroundCluster)
      return getHcalEtSumBc(sc->position(), depth, tower.ieta(), tower.iphi(), -1);
    else if (extIncRule_ == InclusionRule::withinConeAroundCluster and
             intIncRule_ == InclusionRule::isBehindClusterSeed)
      return getHcalEtSumBc(sc->position(), depth, tower.ieta(), tower.iphi(), 1);

    return getHcalEtSumBc(sc->position(), depth, tower.ieta(), tower.iphi(), 0);
  }
  double getHcalESumBc(const math::XYZPoint &p, int depth, int ieta, int iphi, int include_or_exclude) const {
    return getHcalESumBc(GlobalPoint(p.x(), p.y(), p.z()), depth, ieta, iphi, include_or_exclude);
  }
  double getHcalEtSumBc(const math::XYZPoint &p, int depth, int ieta, int iphi, int include_or_exclude) const {
    return getHcalEtSumBc(GlobalPoint(p.x(), p.y(), p.z()), depth, ieta, iphi, include_or_exclude);
  }
  double getHcalESumBc(const GlobalPoint &pclu, int depth, int ieta, int iphi, int include_or_exclude) const {
    return getHcalSum(pclu, depth, ieta, iphi, include_or_exclude, &scaleToE);
  }
  double getHcalEtSumBc(const GlobalPoint &pclu, int depth, int ieta, int iphi, int include_or_exclude) const {
    return getHcalSum(pclu, depth, ieta, iphi, include_or_exclude, &scaleToEt);
  }

private:
  double goodHitEnergy(const GlobalPoint &pclu,
                       const HBHERecHit &hit,
                       int depth,
                       int ieta,
                       int iphi,
                       int include_or_exclude,
                       double (*scale)(const double &)) const;

  double getHcalSum(const GlobalPoint &pclu,
                    int depth,
                    int ieta,
                    int iphi,
                    int include_or_exclude,
                    double (*scale)(const double &)) const;

  InclusionRule extIncRule_;
  double extRadius_;
  InclusionRule intIncRule_;
  double intRadius_;

  arrayHB eThresHB_;
  arrayHB etThresHB_;
  int maxSeverityHB_;

  arrayHE eThresHE_;
  arrayHE etThresHE_;
  int maxSeverityHE_;

  const HBHERecHitCollection &mhbhe_;
  const CaloGeometry &caloGeometry_;
  const HcalTopology &hcalTopology_;
  const HcalChannelQuality &hcalChStatus_;
  const HcalSeverityLevelComputer &hcalSevLvlComputer_;
  const CaloTowerConstituentsMap &towerMap_;
};

#endif
