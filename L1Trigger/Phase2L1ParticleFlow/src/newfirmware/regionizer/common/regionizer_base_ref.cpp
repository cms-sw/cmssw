#include "regionizer_base_ref.h"

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
      float glbEta = sec.region.floatGlbEtaOf(tk), glbPhi = sec.region.floatGlbPhiOf(tk);
      float glbEtaV = sec.region.floatGlbEta(tk.hwVtxEta()), glbPhiV = sec.region.floatGlbPhi(tk.hwVtxPhi());
      for (auto& r : out) {
        if (r.region.contains(glbEta, glbPhi) || r.region.contains(glbEtaV, glbPhiV)) {
          r.track.push_back(tk);
          r.track.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
          r.track.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
        }
      }
    }
  }

  for (const auto& sec : in.hadcalo) {
    for (const auto& c : sec) {
      float glbEta = sec.region.floatGlbEtaOf(c), glbPhi = sec.region.floatGlbPhiOf(c);
      for (auto& r : out) {
        if (r.region.contains(glbEta, glbPhi)) {
          r.hadcalo.push_back(c);
          r.hadcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
          r.hadcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
        }
      }
    }
  }

  for (const auto& sec : in.emcalo) {
    for (const auto& c : sec) {
      float glbEta = sec.region.floatGlbEtaOf(c), glbPhi = sec.region.floatGlbPhiOf(c);
      for (auto& r : out) {
        if (r.region.contains(glbEta, glbPhi)) {
          r.emcalo.push_back(c);
          r.emcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
          r.emcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
        }
      }
    }
  }

  for (const auto& mu : in.muon.obj) {
    float glbEta = mu.floatEta(), glbPhi = mu.floatPhi();
    for (auto& r : out) {
      if (r.region.contains(glbEta, glbPhi)) {
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
