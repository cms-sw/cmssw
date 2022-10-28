#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
l1ct::RegionizerEmulator::RegionizerEmulator(const edm::ParameterSet& iConfig)
    : useAlsoVtxCoords_(iConfig.getParameter<bool>("useAlsoVtxCoords")),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)) {}
#endif

l1ct::RegionizerEmulator::~RegionizerEmulator() {}

void l1ct::RegionizerEmulator::run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) {
  for (const auto& sec : in.track) {
    for (const auto& tk : sec) {
      if (tk.hwPt == 0)
        continue;
      float fglbEta = sec.region.floatGlbEtaOf(tk), fglbPhi = sec.region.floatGlbPhiOf(tk);
      glbeta_t glbEta = sec.region.hwGlbEtaOf(tk);
      glbphi_t glbPhi = sec.region.hwGlbPhiOf(tk);
      glbeta_t glbEtaV = sec.region.hwGlbEta(tk.hwVtxEta());
      glbphi_t glbPhiV = sec.region.hwGlbPhi(tk.hwVtxPhi());
      for (auto& r : out) {
        if (r.region.containsHw(glbEta, glbPhi) || (useAlsoVtxCoords_ && r.region.containsHw(glbEtaV, glbPhiV))) {
          r.track.push_back(tk);
          r.track.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(fglbEta));
          r.track.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(fglbPhi));
        }
      }
    }
  }

  for (const auto& sec : in.hadcalo) {
    for (const auto& c : sec) {
      if (c.hwPt == 0)
        continue;
      float fglbEta = sec.region.floatGlbEtaOf(c), fglbPhi = sec.region.floatGlbPhiOf(c);
      glbeta_t glbEta = sec.region.hwGlbEtaOf(c);
      glbphi_t glbPhi = sec.region.hwGlbPhiOf(c);
      for (auto& r : out) {
        if (r.region.containsHw(glbEta, glbPhi)) {
          r.hadcalo.push_back(c);
          r.hadcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(fglbEta));
          r.hadcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(fglbPhi));
        }
      }
    }
  }

  for (const auto& sec : in.emcalo) {
    for (const auto& c : sec) {
      if (c.hwPt == 0)
        continue;
      float fglbEta = sec.region.floatGlbEtaOf(c), fglbPhi = sec.region.floatGlbPhiOf(c);
      glbeta_t glbEta = sec.region.hwGlbEtaOf(c);
      glbphi_t glbPhi = sec.region.hwGlbPhiOf(c);
      for (auto& r : out) {
        if (r.region.containsHw(glbEta, glbPhi)) {
          r.emcalo.push_back(c);
          r.emcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(fglbEta));
          r.emcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(fglbPhi));
        }
      }
    }
  }

  for (const auto& mu : in.muon.obj) {
    if (mu.hwPt == 0)
      continue;
    float glbEta = mu.floatEta(), glbPhi = mu.floatPhi();
    for (auto& r : out) {
      if (r.region.containsHw(mu.hwEta, mu.hwPhi)) {
        r.muon.push_back(mu);
        r.muon.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
        r.muon.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
      }
    }
  }

  for (auto& r : out) {
    std::sort(r.track.begin(), r.track.end(), [](const l1ct::TkObjEmu& a, const l1ct::TkObjEmu& b) {
      return a.hwPt > b.hwPt;
    });
    std::sort(r.hadcalo.begin(), r.hadcalo.end(), [](const l1ct::HadCaloObjEmu& a, const l1ct::HadCaloObjEmu& b) {
      return a.hwPt > b.hwPt;
    });
    std::sort(r.emcalo.begin(), r.emcalo.end(), [](const l1ct::EmCaloObjEmu& a, const l1ct::EmCaloObjEmu& b) {
      return a.hwPt > b.hwPt;
    });
    std::sort(
        r.muon.begin(), r.muon.end(), [](const l1ct::MuObjEmu& a, const l1ct::MuObjEmu& b) { return a.hwPt > b.hwPt; });
  }
}
