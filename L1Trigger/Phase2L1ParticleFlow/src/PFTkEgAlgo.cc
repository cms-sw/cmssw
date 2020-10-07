#include "L1Trigger/Phase2L1ParticleFlow/interface/PFTkEGAlgo.h"

using namespace l1tpf_impl;

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <algorithm>

namespace {
  template <typename T1, typename T2>
  float floatDR(const T1 &t1, const T2 &t2) {
    return deltaR(t1.floatEta(), t1.floatPhi(), t2.floatEta(), t2.floatPhi());
  }
}  // namespace

PFTkEGAlgo::PFTkEGAlgo(const edm::ParameterSet &) {}

PFTkEGAlgo::~PFTkEGAlgo() {}

void PFTkEGAlgo::initRegion(Region &r) const {
  // FIXME: assume imput is sorted already
  r.egobjs.clear();
}

void PFTkEGAlgo::runTkEG(Region &r) const {
  initRegion(r);

  //FIXME: configurable (off for barrel)
  bool doBremRecovery = true;

  // NOTE: we run this step for all clusters (before matching) as it is done in the pre-PF algorithm
  std::vector<int> emCalo2emCalo(r.emcalo.size(), -1);
  if (doBremRecovery)
    link_emCalo2emCalo(r, emCalo2emCalo);

  // track to EM calo matching
  std::vector<int> emCalo2tk(r.emcalo.size(), -1);
  link_emCalo2tk(r, emCalo2tk);

  //FIXME: compute tk based iso

  // add tk electron to region;
  eg_algo(r, emCalo2emCalo, emCalo2tk);
}

void PFTkEGAlgo::eg_algo(Region &r, const std::vector<int> &emCalo2emCalo, const std::vector<int> &emCalo2tk) const {
  // FIXME: configurables
  int caloHwQual_ = 4;

  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (calo.hwFlags != caloHwQual_)
      continue;

    int itk = emCalo2tk[ic];
    // 1. create EG objects before brem recovery
    addEgObjsToPF(r.egobjs, ic, calo.hwFlags, calo.floatPt(), itk);

    // check if the cluster has already been used in a brem reclustering
    if (emCalo2emCalo[ic] != -1)
      continue;

    float ptBremReco = calo.floatPt();

    for (int jc = ic; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] == ic) {
        auto &otherCalo = r.emcalo[jc];
        ptBremReco += otherCalo.floatPt();
      }
    }

    // 2. create TkEle with brem recovery
    addEgObjsToPF(r.egobjs, ic, calo.hwFlags + 1, ptBremReco, itk);
  }
}

l1tpf_impl::EgObjectIndexer &PFTkEGAlgo::addEgObjsToPF(std::vector<l1tpf_impl::EgObjectIndexer> egobjs,
                                                       const int calo_idx,
                                                       const int hwQual,
                                                       const float ptCorr,
                                                       const int tk_idx,
                                                       const float iso) const {
  egobjs.emplace_back(l1tpf_impl::EgObjectIndexer{calo_idx, hwQual, ptCorr, tk_idx, iso});
  return egobjs.back();
}

void PFTkEGAlgo::link_emCalo2emCalo(Region &r, std::vector<int> &emCalo2emCalo) const {
  // FIXME: configurables
  int caloHwQual_ = 4;
  float dEtaMax = 0.02;
  float dPhiMax = 0.1;

  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (calo.hwFlags != caloHwQual_)
      continue;

    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = r.emcalo[jc];
      if (fabs(otherCalo.floatEta() - calo.floatEta()) < dEtaMax &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < dPhiMax) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}

void PFTkEGAlgo::link_emCalo2tk(Region &r, std::vector<int> &emCalo2tk) const {
  // FIXME: configurable
  // configuration of the elliptic cut over the whole detector
  std::vector<float> absEtaBountaries_{0.0, 0.9, 1.5};
  std::vector<float> dEtaValues_{0.025, 0.015, 0.01};  // last was  0.0075  in TDR
  std::vector<float> dPhiValues_{0.07, 0.07, 0.07};
  float caloEtMin_ = 0.0;
  // FIXME this needs configuration depending on where we are...? Or maybe a list of qualities
  int caloHwQual_ = 4;

  float trkQualityPtMin_ = 10.0;
  float trkQualityChi2_ = 1e10;

  // track to calo matching (first iteration, with a lower bound on the calo pt; there may be another one later)
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    // std::cout << "[" << ic << "] pt: " << calo.floatPt() << std::endl;
    if (calo.hwFlags != caloHwQual_)
      continue;
    // compute elliptic matching
    auto eta_index = std::distance(
        absEtaBountaries_.begin(),
        std::lower_bound(absEtaBountaries_.begin(), absEtaBountaries_.end(), r.globalAbsEta(calo.floatEta())));
    float dEtaMax = dEtaValues_[eta_index];
    float dPhiMax = dPhiValues_[eta_index];

    float dPtMin = 999;

    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      if (tk.floatPt() < trkQualityPtMin_)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = fabs(tk.floatEta() - calo.floatEta());

      // std::cout << "Global abs eta: " << r.globalAbsEta(calo.floatEta())
      //           << " abs eta: " << fabs(calo.floatEta()) << std::endl;

      if (((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax)) < 1.) {
        // FIXME: how to define the best match? Closest in pt? See what done in PF
        // maybe we want the highest in pT and then recover using the brem recovery???
        if (fabs(tk.floatPt() - calo.floatPt()) < dPtMin) {
          emCalo2tk[ic] = itk;
          dPtMin = fabs(tk.floatPt() - calo.floatPt());
        }
      }
    }
  }
}
