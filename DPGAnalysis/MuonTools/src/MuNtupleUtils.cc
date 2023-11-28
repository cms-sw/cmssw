/*
 * \file MuNtupleUtils.cc
 *
 * \author C. Battilana - INFN (BO)
 * \author L. Giuducci - INFN (BO)
*/

#include "DPGAnalysis/MuonTools/interface/MuNtupleUtils.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

// CB
// formulas to be re-checked
// can use template is_same, static_assert, enable_if ...

nano_mu::DTTrigGeomUtils::DTTrigGeomUtils(edm::ConsumesCollector&& collector, bool dirInDeg)
    : m_dtGeom{std::move(collector), "idealForDigi"} {}

nano_mu::DTTrigGeomUtils::chambCoord nano_mu::DTTrigGeomUtils::trigToReco(const L1MuDTChambPhDigi* trig) {
  auto wh{trig->whNum()};
  auto sec{trig->scNum() + 1};
  auto st{trig->stNum()};
  auto phi{trig->phi()};
  auto phib{trig->phiB()};

  auto recoChamb = [&]() {
    if (st != 4) {
      return DTChamberId(wh, st, sec);
    }
    int reco_sec{(sec == 4 && phi > 0) ? 13 : (sec == 10 && phi > 0) ? 14 : sec};
    return DTChamberId(wh, st, reco_sec);
  };

  auto gpos{m_dtGeom->chamber(recoChamb())->position()};
  auto r{gpos.perp()};

  auto delta_phi{gpos.phi() - (sec - 1) * Geom::pi() / 6};

  // zcn is in local coordinates -> z invreases approching to vertex
  // LG: zcn offset was removed <- CB do we need to fix this?
  float x = r * tan((phi - (phi < 0 ? 1 : 0)) / PH1_PHI_R) * cos(delta_phi) - r * sin(delta_phi);
  float dir = (phib / PH1_PHIB_R + phi / PH1_PHI_R);

  // change sign in case of positive wheels
  if (hasPosRF(wh, sec)) {
    x = -x;
  } else {
    dir = -dir;
  }

  return {x, dir};
}

nano_mu::DTTrigGeomUtils::chambCoord nano_mu::DTTrigGeomUtils::trigToReco(const L1Phase2MuDTPhDigi* trig) {
  auto wh{trig->whNum()};
  auto sec{trig->scNum() + 1};
  auto st{trig->stNum()};
  auto phi{trig->phi()};
  auto phib{trig->phiBend()};
  auto quality{trig->quality()};
  auto sl{trig->slNum()};

  auto recoChamb = [&]() {
    if (st != 4) {
      return DTChamberId(wh, st, sec);
    }
    int reco_sec{(sec == 4 && phi > 0) ? 13 : (sec == 10 && phi > 0) ? 14 : sec};
    return DTChamberId(wh, st, reco_sec);
  };

  auto gpos{m_dtGeom->chamber(recoChamb())->position()};
  auto r{gpos.perp()};

  auto delta_phi{gpos.phi() - (sec - 1) * Geom::pi() / 6};

  // CB to be potentially updated based on Silvia's results
  double zRF = 0;
  if (quality >= 6 && quality != 7)
    zRF = m_zcn[st - 1];
  if ((quality < 6 || quality == 7) && sl == 1)
    zRF = m_zsl1[st - 1];
  if ((quality < 6 || quality == 7) && sl == 3)
    zRF = m_zsl3[st - 1];

  // zcn is in local coordinates -> z invreases approching to vertex
  // LG: zcn offset was removed <- CB Mist confirm it is tryly accurate?
  float x = r * tan((phi - (phi < 0 ? 1 : 0)) / PH1_PHI_R) * (cos(delta_phi) - zRF) - r * sin(delta_phi);
  float dir = (phib / PH2_PHIB_R + phi / PH2_PHI_R);

  // change sign in case of positive wheels
  if (hasPosRF(wh, sec)) {
    x = -x;
  } else {
    dir = -dir;
  }

  return {x, dir};
}
