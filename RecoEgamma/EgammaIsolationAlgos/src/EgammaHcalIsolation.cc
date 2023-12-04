//*****************************************************************************
// File:      EgammaHcalIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"

double scaleToE(const double &eta) { return 1.; }
double scaleToEt(const double &eta) { return std::sin(2. * std::atan(std::exp(-eta))); }

EgammaHcalIsolation::EgammaHcalIsolation(InclusionRule extIncRule,
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
                                         edm::ESHandle<CaloTowerConstituentsMap> towerMap)
    : extIncRule_(extIncRule),
      extRadius_(extRadius * extRadius),
      intIncRule_(intIncRule),
      intRadius_(intRadius * intRadius),
      maxSeverityHB_(maxSeverityHB),
      maxSeverityHE_(maxSeverityHE),
      mhbhe_(mhbhe),
      caloGeometry_(*caloGeometry.product()),
      hcalTopology_(*hcalTopology.product()),
      hcalChStatus_(*hcalChStatus.product()),
      hcalSevLvlComputer_(*hcalSevLvlComputer.product()),
      towerMap_(*towerMap.product()) {
  eThresHB_ = eThresHB;
  etThresHB_ = etThresHB;
  eThresHE_ = eThresHE;
  etThresHE_ = etThresHE;

  // make some adjustments for the BC rules
  if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::withinConeAroundCluster) {
    extRadius_ = 0.;
    intRadius_ = 0.;
  } else if (extIncRule_ == InclusionRule::withinConeAroundCluster and
             intIncRule_ == InclusionRule::isBehindClusterSeed) {
    intRadius_ = 0.;
  } else if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::isBehindClusterSeed) {
    edm::LogWarning("EgammaHcalIsolation")
        << " external and internal rechit inclusion rules can't both be isBehindClusterSeed."
        << " Setting both to withinConeAroundCluster!";
    extIncRule_ = InclusionRule::withinConeAroundCluster;
    intIncRule_ = InclusionRule::withinConeAroundCluster;
  }
}

EgammaHcalIsolation::EgammaHcalIsolation(InclusionRule extIncRule,
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
                                         const CaloTowerConstituentsMap &towerMap)
    : extIncRule_(extIncRule),
      extRadius_(extRadius * extRadius),
      intIncRule_(intIncRule),
      intRadius_(intRadius * intRadius),
      maxSeverityHB_(maxSeverityHB),
      maxSeverityHE_(maxSeverityHE),
      mhbhe_(mhbhe),
      caloGeometry_(caloGeometry),
      hcalTopology_(hcalTopology),
      hcalChStatus_(hcalChStatus),
      hcalSevLvlComputer_(hcalSevLvlComputer),
      towerMap_(towerMap) {
  eThresHB_ = eThresHB;
  etThresHB_ = etThresHB;
  eThresHE_ = eThresHE;
  etThresHE_ = etThresHE;

  // make some adjustments for the BC rules
  if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::withinConeAroundCluster) {
    extRadius_ = 0.;
    intRadius_ = 0.;
  } else if (extIncRule_ == InclusionRule::withinConeAroundCluster and
             intIncRule_ == InclusionRule::isBehindClusterSeed) {
    intRadius_ = 0.;
  } else if (extIncRule_ == InclusionRule::isBehindClusterSeed and intIncRule_ == InclusionRule::isBehindClusterSeed) {
    edm::LogWarning("EgammaHcalIsolation")
        << " external and internal rechit inclusion rules can't both be isBehindClusterSeed."
        << " Setting both to withinConeAroundCluster!";
    extIncRule_ = InclusionRule::withinConeAroundCluster;
    intIncRule_ = InclusionRule::withinConeAroundCluster;
  }
}

double EgammaHcalIsolation::goodHitEnergy(float pcluEta,
                                          float pcluPhi,
                                          const HBHERecHit &hit,
                                          int depth,
                                          int ieta,
                                          int iphi,
                                          int include_or_exclude,
                                          double (*scale)(const double &),
                                          const HcalPFCuts *hcalCuts) const {
  const HcalDetId hid(hit.detid());
  const int hd = hid.depth(), he = hid.ieta(), hp = hid.iphi();
  const int h1 = hd - 1;
  double thresholdE = 0.;

  if (include_or_exclude == -1 and (he != ieta or hp != iphi))
    return 0.;

  if (include_or_exclude == 1 and (he == ieta and hp == iphi))
    return 0.;

  if ((hid.subdet() == HcalBarrel and (hd < 1 or hd > int(eThresHB_.size()))) or
      (hid.subdet() == HcalEndcap and (hd < 1 or hd > int(eThresHE_.size()))))
    edm::LogWarning("EgammaHcalIsolation")
        << " hit in subdet " << hid.subdet() << " has an unaccounted for depth of " << hd << "!!";

  const bool right_depth = (depth == 0 or hd == depth);
  if (!right_depth)
    return 0.;

  bool goodHBe = hid.subdet() == HcalBarrel and hit.energy() > eThresHB_[h1];
  bool goodHEe = hid.subdet() == HcalEndcap and hit.energy() > eThresHE_[h1];

  if (hcalCuts != nullptr) {
    const HcalPFCut *cutValue = hcalCuts->getValues(hid.rawId());
    thresholdE = cutValue->noiseThreshold();
    goodHBe = hid.subdet() == HcalBarrel and hit.energy() > thresholdE;
    goodHEe = hid.subdet() == HcalEndcap and hit.energy() > thresholdE;
  }

  if (!(goodHBe or goodHEe))
    return 0.;

  const auto phit = caloGeometry_.getGeometry(hit.detid())->repPos();
  const float phitEta = phit.eta();

  if (extIncRule_ == InclusionRule::withinConeAroundCluster or intIncRule_ == InclusionRule::withinConeAroundCluster) {
    auto const dR2 = deltaR2(pcluEta, pcluPhi, phitEta, phit.phi());
    if ((extIncRule_ == InclusionRule::withinConeAroundCluster and dR2 > extRadius_) or
        (intIncRule_ == InclusionRule::withinConeAroundCluster and dR2 < intRadius_))
      return 0.;
  }

  DetId did = hcalTopology_.idFront(hid);
  const uint32_t flag = hit.flags();
  const uint32_t dbflag = hcalChStatus_.getValues(did)->getValue();
  int severity = hcalSevLvlComputer_.getSeverityLevel(did, flag, dbflag);
  bool recovered = hcalSevLvlComputer_.recoveredRecHit(did, flag);

  const double het = hit.energy() * scaleToEt(phitEta);
  const bool goodHB = goodHBe and (severity <= maxSeverityHB_ or recovered) and het > etThresHB_[h1];
  const bool goodHE = goodHEe and (severity <= maxSeverityHE_ or recovered) and het > etThresHE_[h1];

  if (goodHB or goodHE)
    return hit.energy() * scale(phitEta);

  return 0.;
}

double EgammaHcalIsolation::getHcalSum(const GlobalPoint &pclu,
                                       int depth,
                                       int ieta,
                                       int iphi,
                                       int include_or_exclude,
                                       double (*scale)(const double &),
                                       const HcalPFCuts *hcalCuts) const {
  double sum = 0.;
  const float pcluEta = pclu.eta();
  const float pcluPhi = pclu.phi();
  for (const auto &hit : mhbhe_)
    sum += goodHitEnergy(pcluEta, pcluPhi, hit, depth, ieta, iphi, include_or_exclude, scale, hcalCuts);

  return sum;
}
