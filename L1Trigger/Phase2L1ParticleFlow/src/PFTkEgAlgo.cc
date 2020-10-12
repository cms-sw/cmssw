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

PFTkEGAlgo::PFTkEGAlgo(const edm::ParameterSet &pset):
  debug_(pset.getUntrackedParameter<int>("debug")),
  doBremRecovery_(pset.getParameter<bool>("doBremRecovery")),
  filterHwQuality_(pset.getParameter<bool>("filterHwQuality")),
  caloHwQual_(pset.getParameter<int>("caloHwQual")),
  dEtaMaxBrem_(pset.getParameter<double>("dEtaMaxBrem")),
  dPhiMaxBrem_(pset.getParameter<double>("dPhiMaxBrem")),
  absEtaBoundaries_(pset.getParameter<std::vector<double>>("absEtaBoundaries")),
  dEtaValues_(pset.getParameter<std::vector<double>>("dEtaValues")),
  dPhiValues_(pset.getParameter<std::vector<double>>("dPhiValues")),
  caloEtMin_(pset.getParameter<double>("caloEtMin")),
  trkQualityPtMin_(pset.getParameter<double>("trkQualityPtMin")),
  trkQualityChi2_(pset.getParameter<double>("trkQualityChi2")),
  writeEgSta_(pset.getParameter<bool>("writeEgSta")) {}

PFTkEGAlgo::~PFTkEGAlgo() {}

void PFTkEGAlgo::initRegion(Region &r) const {
  // FIXME: assume imput is sorted already
  r.egobjs.clear();
}

void PFTkEGAlgo::runTkEG(Region &r) const {
  initRegion(r);

  if(debug_ > 0) {
    std::cout << "[PFTkEGAlgo::runTkEG] START" << std::endl;
    std::cout << "   # emCalo: " << r.emcalo.size() << " # tk: " << r.track.size() << std::endl;
  }
  // NOTE: we run this step for all clusters (before matching) as it is done in the pre-PF EG algorithm
  std::vector<int> emCalo2emCalo(r.emcalo.size(), -1);
  if (doBremRecovery_)
    link_emCalo2emCalo(r, emCalo2emCalo);

  // track to EM calo matching
  std::vector<int> emCalo2tk(r.emcalo.size(), -1);
  link_emCalo2tk(r, emCalo2tk);

  //FIXME: compute tk based iso

  // add tk electron to region;
  eg_algo(r, emCalo2emCalo, emCalo2tk);
}

void PFTkEGAlgo::eg_algo(Region &r, const std::vector<int> &emCalo2emCalo, const std::vector<int> &emCalo2tk) const {

  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;

    int itk = emCalo2tk[ic];
    // 1. create EG objects before brem recovery
    addEgObjsToPF(r.egobjs, ic, calo.hwFlags, calo.floatPt(), itk);

    // check if brem recovery is on
    if(!doBremRecovery_)
      continue;

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

    // 2. create EG objects with brem recovery
    addEgObjsToPF(r.egobjs, ic, calo.hwFlags + 1, ptBremReco, itk);
  }
}

l1tpf_impl::EgObjectIndexer &PFTkEGAlgo::addEgObjsToPF(std::vector<l1tpf_impl::EgObjectIndexer> &egobjs,
                                                       const int calo_idx,
                                                       const int hwQual,
                                                       const float ptCorr,
                                                       const int tk_idx,
                                                       const float iso) const {
  egobjs.emplace_back(l1tpf_impl::EgObjectIndexer{calo_idx, hwQual, ptCorr, tk_idx, iso});
  return egobjs.back();
}

void PFTkEGAlgo::link_emCalo2emCalo(Region &r, std::vector<int> &emCalo2emCalo) const {
  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;

    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic+1; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = r.emcalo[jc];
      if (filterHwQuality_ && otherCalo.hwFlags != caloHwQual_)
        continue;

      if (fabs(otherCalo.floatEta() - calo.floatEta()) < dEtaMaxBrem_ &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < dEtaMaxBrem_) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}

void PFTkEGAlgo::link_emCalo2tk(Region &r, std::vector<int> &emCalo2tk) const {
  // track to calo matching (first iteration, with a lower bound on the calo pt; there may be another one later)
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];

    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;
    // compute elliptic matching
    auto eta_index = std::distance(
        absEtaBoundaries_.begin(),
        std::lower_bound(absEtaBoundaries_.begin(), absEtaBoundaries_.end(), r.globalAbsEta(calo.floatEta()))) - 1;
    float dEtaMax = dEtaValues_[eta_index];
    float dPhiMax = dPhiValues_[eta_index];

    if(debug_ > 0) std::cout << "idx: " << eta_index << " deta: " << dEtaMax << " dphi: " << dPhiMax << std::endl;

    float dPtMin = 999;
    if(debug_ > 0) std::cout << "--- calo: pt: " << calo.floatPt() << " eta: " << calo.floatEta() << " phi: " << calo.floatPhi() << std::endl;
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      if(debug_ > 0) std::cout << "  - tk: pt: " << tk.floatPt() << " eta: " << tk.floatEta() << " phi: " << tk.floatPhi() << std::endl;

      if (tk.floatPt() < trkQualityPtMin_)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = fabs(tk.floatEta() - calo.floatEta());

      if(debug_ > 0) std::cout << " deta: " << d_eta << " dphi: " << d_phi << " ell: " << ((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))<< std::endl;
      // std::cout << "Global abs eta: " << r.globalAbsEta(calo.floatEta())
      //           << " abs eta: " << fabs(calo.floatEta()) << std::endl;

      if ((((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) < 1.) {
        if(debug_ > 0) std::cout << "    pass elliptic " << std::endl;
        // NOTE: for now we implement only best pt match. This is NOT what is done in the L1TkElectronTrackProducer
        if (fabs(tk.floatPt() - calo.floatPt()) < dPtMin) {
          if(debug_ > 0) std::cout << "     best pt match: " << fabs(tk.floatPt() - calo.floatPt()) << std::endl;
          emCalo2tk[ic] = itk;
          dPtMin = fabs(tk.floatPt() - calo.floatPt());
        }
      }
    }
  }
}
