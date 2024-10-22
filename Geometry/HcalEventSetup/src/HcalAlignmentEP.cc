// -*- C++ -*-
//
// Package:    HcalAlignmentEP
// Class:      HcalAlignmentEP
//
//
// Original Author:  Brian Heltsley
//
//

#include "Geometry/HcalEventSetup/interface/HcalAlignmentEP.h"

HcalAlignmentEP::HcalAlignmentEP(const edm::ParameterSet&) {
  auto cc = setWhatProduced(this, &HcalAlignmentEP::produceHcalAli);
  hbToken_ = cc.consumesFrom<Alignments, HBAlignmentRcd>(edm::ESInputTag{});
  heToken_ = cc.consumesFrom<Alignments, HEAlignmentRcd>(edm::ESInputTag{});
  hfToken_ = cc.consumesFrom<Alignments, HFAlignmentRcd>(edm::ESInputTag{});
  hoToken_ = cc.consumesFrom<Alignments, HOAlignmentRcd>(edm::ESInputTag{});
}

HcalAlignmentEP::~HcalAlignmentEP() {}

HcalAlignmentEP::ReturnAli HcalAlignmentEP::produceHcalAli(const HcalAlignmentRcd& iRecord) {
  auto ali = std::make_unique<Alignments>();

  std::vector<AlignTransform>& vtr(ali->m_align);
  const unsigned int nA(HcalGeometry::numberOfAlignments());
  vtr.resize(nA);

  const auto& hb = iRecord.get(hbToken_);
  const auto& he = iRecord.get(heToken_);
  const auto& hf = iRecord.get(hfToken_);
  const auto& ho = iRecord.get(hoToken_);

  // require valid alignments and expected size
  assert(hb.m_align.size() == HcalGeometry::numberOfBarrelAlignments());
  assert(he.m_align.size() == HcalGeometry::numberOfEndcapAlignments());
  assert(hf.m_align.size() == HcalGeometry::numberOfForwardAlignments());
  assert(ho.m_align.size() == HcalGeometry::numberOfOuterAlignments());
  const std::vector<AlignTransform>& hbt = hb.m_align;
  const std::vector<AlignTransform>& het = he.m_align;
  const std::vector<AlignTransform>& hft = hf.m_align;
  const std::vector<AlignTransform>& hot = ho.m_align;

  copy(hbt.begin(), hbt.end(), vtr.begin());
  copy(het.begin(), het.end(), vtr.begin() + hbt.size());
  copy(hft.begin(), hft.end(), vtr.begin() + hbt.size() + het.size());
  copy(hot.begin(), hot.end(), vtr.begin() + hbt.size() + het.size() + hft.size());

  return ali;
}
