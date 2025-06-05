#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

L1SCJetEmu::L1SCJetEmu(bool debug, float coneSize, unsigned nJets)
    : debug_(debug),
      coneSize_(coneSize),
      nJets_(nJets),
      rCone2_(coneSize * coneSize / l1ct::Scales::ETAPHI_LSB / l1ct::Scales::ETAPHI_LSB) {
  init_invert_table<pt_t, inv_pt_t, N_table_inv_pt>(inv_pt_table_);
}

L1SCJetEmu::detaphi_t L1SCJetEmu::deltaPhi(L1SCJetEmu::Particle a, L1SCJetEmu::Particle b) {
  detaphi_t dphi = detaphi_t(a.hwPhi) - detaphi_t(b.hwPhi);
  // phi wrap
  detaphi_t dphi0 =
      dphi > detaphi_t(l1ct::Scales::INTPHI_PI) ? detaphi_t(l1ct::Scales::INTPHI_TWOPI - dphi) : detaphi_t(dphi);
  detaphi_t dphi1 =
      dphi < detaphi_t(-l1ct::Scales::INTPHI_PI) ? detaphi_t(l1ct::Scales::INTPHI_TWOPI + dphi) : detaphi_t(dphi);
  detaphi_t dphiw = dphi > detaphi_t(0) ? dphi0 : dphi1;
  return dphiw;
}

bool L1SCJetEmu::inCone(L1SCJetEmu::Particle seed, L1SCJetEmu::Particle part) const {
  // scale the particle eta, phi to hardware units
  detaphi_t deta = detaphi_t(seed.hwEta) - detaphi_t(part.hwEta);
  detaphi_t dphi = deltaPhi(seed, part);
  bool ret = deta * deta + dphi * dphi < rCone2_;
  //bool ret = r2 < cone2;
  if (debug_) {
    detaphi2_t r2 = detaphi2_t(deta) * detaphi2_t(deta) + detaphi2_t(dphi) * detaphi2_t(dphi);
    dbgCout() << "  part eta, seed eta: " << part.hwEta << ", " << seed.hwEta << std::endl;
    dbgCout() << "  part phi, seed phi: " << part.hwPhi << ", " << seed.hwPhi << std::endl;
    dbgCout() << "  pt, deta, dphi, r2, cone2, lt: " << part.hwPt << ", " << deta << ", " << dphi << ", "
              << deta * deta + dphi * dphi << ", " << rCone2_ << ", " << ret << std::endl;
  }
  return ret;
}

std::vector<L1SCJetEmu::Particle> L1SCJetEmu::sortConstituents(const std::vector<Particle>& parts,
                                                               const Particle seed) const {
  std::vector<Particle> sortedParts = parts;  // instantiate a vector to store sorted parts
  std::sort(sortedParts.begin(), sortedParts.end(), [](const Particle& a, const Particle& b) {
    return a.hwPt > b.hwPt;
  });                               // sort by pt by jet mass fn as in firmware
  std::vector<Particle> truncated;  // instantiate vector to store truncated, sorted parts
  truncated.resize(NCONSTITSFW);
  for (unsigned iConst = 0; iConst < NCONSTITSFW; ++iConst) {  // iterate over NCONSTITS (or truncated.size())
    if (iConst <
        sortedParts.size()) {  // if iConst is less than the number of constituents in the jet then store the constituent
      truncated[iConst].hwEta = static_cast<detaphi_t>(sortedParts.at(iConst).hwEta - seed.hwEta);
      truncated[iConst].hwPhi = static_cast<detaphi_t>(deltaPhi(sortedParts.at(iConst), seed));
      truncated[iConst].hwPt = sortedParts.at(iConst).hwPt;
    } else {  // if iConst is greater than the number of constituents in the jet then store an empty constituent (pt = 0) to mimic sparse array from firmware
      truncated[iConst].clear();
    }
  }
  return truncated;
}

L1SCJetEmu::mass2_t L1SCJetEmu::jetMass_HW(const std::vector<Particle>& parts) const {  // need ampersand?

  // // INSTANTIATE LUTS
  static std::array<eventrig_t, hwEtaPhi_steps> cosh_lut =
      init_trig_lut<eventrig_t, hwEtaPhi_steps>([](float x) -> eventrig_t { return std::cosh(x); });
  static std::array<eventrig_t, hwEtaPhi_steps> cos_lut =
      init_trig_lut<eventrig_t, hwEtaPhi_steps>([](float x) -> eventrig_t { return std::cos(x); });
  static std::array<oddtrig_t, hwEtaPhi_steps> sin_lut =
      init_trig_lut<oddtrig_t, hwEtaPhi_steps>([](float x) -> oddtrig_t { return std::sin(x); });
  static std::array<oddtrig_t, hwEtaPhi_steps> sinh_lut =
      init_trig_lut<oddtrig_t, hwEtaPhi_steps>([](float x) -> oddtrig_t { return std::sinh(x); });

  std::vector<ppt_t> en;
  en.resize(parts.size());
  std::transform(parts.begin(), parts.end(), en.begin(), [](const Particle& part) {
    return ppt_t(part.hwPt * cosh_lut[std::abs(part.hwEta)]);
  });
  ppt_t sum_en = std::accumulate(en.begin(), en.end(), ppt_t(0));

  std::vector<ppt_t> px;
  px.resize(parts.size());
  std::transform(parts.begin(), parts.end(), px.begin(), [](const Particle& part) {
    return ppt_t(part.hwPt * cos_lut[std::abs(part.hwPhi)]);
  });
  ppt_t sum_px = std::accumulate(px.begin(), px.end(), ppt_t(0));

  std::vector<npt_t> py;
  py.resize(parts.size());
  std::transform(parts.begin(), parts.end(), py.begin(), [](const Particle& part) {
    return npt_t(part.hwPt * sin_lut[std::abs(part.hwPhi)] * ((part.hwPhi >= 0) ? 1 : -1));
  });
  npt_t sum_py = std::accumulate(py.begin(), py.end(), npt_t(0));

  std::vector<npt_t> pz;
  pz.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pz.begin(), [](const Particle& part) {
    return npt_t(part.hwPt * sinh_lut[std::abs(part.hwEta)] * ((part.hwEta >= 0) ? 1 : -1));
  });
  npt_t sum_pz = std::accumulate(pz.begin(), pz.end(), npt_t(0));

  return (sum_en * sum_en) - (sum_px * sum_px) - (sum_py * sum_py) - (sum_pz * sum_pz);
}

L1SCJetEmu::Jet L1SCJetEmu::makeJet_HW(const std::vector<Particle>& parts, const Particle seed) const {
  // Seed Cone Jet algorithm with ap_fixed types and hardware emulation

  // Event with saturation, order of terms doesn't matter since they're all positive
  auto sumpt = [](pt_t(a), const Particle& b) { return a + b.hwPt; };  // essentially a python lambda fn

  // Sum the pt
  pt_t pt = std::accumulate(parts.begin(), parts.end(), pt_t(0), sumpt);
  inv_pt_t inv_pt = invert_with_shift<pt_t, inv_pt_t, N_table_inv_pt>(pt, inv_pt_table_, false);

  // pt weighted d eta
  std::vector<pt_etaphi_t> pt_deta;
  pt_deta.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_deta.begin(), [&seed](const Particle& part) {
    // In the firmware we calculate the per-particle pt-weighted deta
    return pt_etaphi_t(part.hwPt * detaphi_t(part.hwEta - seed.hwEta));
  });
  // Accumulate the pt-weighted etas. Init to 0, include seed in accumulation
  pt_etaphi_t sum_pt_eta = std::accumulate(pt_deta.begin(), pt_deta.end(), pt_etaphi_t(0));
  etaphi_t eta = seed.hwEta + etaphi_t(sum_pt_eta * inv_pt);

  // pt weighted d phi
  std::vector<pt_etaphi_t> pt_dphi;
  pt_dphi.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_dphi.begin(), [&seed](const Particle& part) {
    // In the firmware we calculate the per-particle pt-weighted dphi
    return pt_etaphi_t(part.hwPt * deltaPhi(part, seed));
  });
  // Accumulate the pt-weighted phis. Init to 0, include seed in accumulation
  pt_etaphi_t sum_pt_phi = std::accumulate(pt_dphi.begin(), pt_dphi.end(), pt_etaphi_t(0));
  etaphi_t phi = seed.hwPhi + etaphi_t(sum_pt_phi * inv_pt);  // shift the seed by pt weighted sum_pt_phi

  std::vector<Particle> truncated =
      sortConstituents(parts, seed);  // sort the constituents by pt and truncate to NCONSTITS
  mass2_t massSq = L1SCJetEmu::jetMass_HW(truncated);

  Jet jet;
  jet.hwPt = pt;
  jet.hwEta = eta;
  jet.hwPhi = phi;
  jet.hwMassSq = massSq;
  jet.constituents = parts;
  // jet.constituents = truncated;  // store the truncated, sorted NCONSTITSFW sparse array of constituents

  if (debug_) {
    std::for_each(pt_dphi.begin(), pt_dphi.end(), [](pt_etaphi_t& x) { dbgCout() << "pt_dphi: " << x << std::endl; });
    std::for_each(pt_deta.begin(), pt_deta.end(), [](pt_etaphi_t& x) { dbgCout() << "pt_deta: " << x << std::endl; });
    dbgCout() << " sum_pt_eta: " << sum_pt_eta << ", 1/pt: " << inv_pt
              << ", sum_pt_eta * 1/pt: " << etaphi_t(sum_pt_eta * inv_pt) << std::endl;
    dbgCout() << " sum_pt_phi: " << sum_pt_phi << ", 1/pt: " << inv_pt
              << ", sum_pt_phi * 1/pt: " << etaphi_t(sum_pt_phi * inv_pt) << std::endl;
    dbgCout() << " uncorr eta: " << seed.hwEta << ", phi: " << seed.hwPhi << std::endl;
    dbgCout() << "   corr eta: " << eta << ", phi: " << phi << std::endl;
    dbgCout() << "         pt: " << pt << std::endl;
  }

  return jet;
}

std::vector<L1SCJetEmu::Jet> L1SCJetEmu::emulateEvent(std::vector<Particle>& parts) const {
  // The fixed point algorithm emulation
  std::vector<Particle> work;
  work.resize(parts.size());
  std::transform(parts.begin(), parts.end(), work.begin(), [](const Particle& part) { return part; });

  std::vector<Jet> jets;
  jets.reserve(nJets_);
  while ((!work.empty() && jets.size() < nJets_)) {
    // Take the highest pt candidate as a seed
    // Use the firmware reduce function to find the same seed as the firmware
    // in case there are multiple seeds with the same pT
    // ... or use external seed if configured to do so
    Particle seed = reduce(work, op_max);

    // Get the particles within a coneSize_ of the seed
    std::vector<Particle> particlesInCone;
    std::copy_if(work.begin(), work.end(), std::back_inserter(particlesInCone), [&](const Particle& part) {
      return inCone(seed, part);
    });
    if (debug_) {
      dbgCout() << "Seed: " << seed.hwPt << ", " << seed.hwEta << ", " << seed.hwPhi << std::endl;
      dbgCout() << "N particles : " << particlesInCone.size() << std::endl;
      std::for_each(particlesInCone.begin(), particlesInCone.end(), [&](Particle& part) {
        dbgCout() << "  Part: " << part.hwPt << ", " << part.hwEta << ", " << part.hwPhi << std::endl;
        inCone(seed, part);
      });
    }
    jets.push_back(makeJet_HW(particlesInCone, seed));
    //remove the clustered particles
    work.erase(std::remove_if(work.begin(),
                              work.end(),
                              [&](const Particle& part) {
                                return inCone(seed, part);
                              }),  //erase particles from further jet clustering
               work.end());
  }
  return jets;
}
