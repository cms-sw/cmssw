#include "L1Trigger/Phase2L1ParticleFlow/interface/PFTkEGAlgo.h"

using namespace l1tpf_impl;

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <algorithm>
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  template <typename T1, typename T2>
  float floatDR(const T1 &t1, const T2 &t2) {
    return deltaR(t1.floatEta(), t1.floatPhi(), t2.floatEta(), t2.floatPhi());
  }
}  // namespace

PFTkEGAlgo::PFTkEGAlgo(const edm::ParameterSet &pset)
    : debug_(pset.getUntrackedParameter<int>("debug")),
      doBremRecovery_(pset.getParameter<bool>("doBremRecovery")),
      doTkIsolation_(pset.getParameter<bool>("doTkIsolation")),
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
      writeEgSta_(pset.getParameter<bool>("writeEgSta")),
      tkIsoParametersTkEm_(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEm")),
      tkIsoParametersTkEle_(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEle")),
      pfIsoParametersTkEm_(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEm")),
      pfIsoParametersTkEle_(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEle")) {}

PFTkEGAlgo::~PFTkEGAlgo() {}

PFTkEGAlgo::IsoParameters::IsoParameters(const edm::ParameterSet &pset)
    : tkQualityPtMin(pset.getParameter<double>("tkQualityPtMin")),
      dZ(pset.getParameter<double>("dZ")),
      dRMin(pset.getParameter<double>("dRMin")),
      dRMax(pset.getParameter<double>("dRMax")),
      tkQualityChi2Max(pset.getParameter<double>("tkQualityChi2Max")),
      dRMin2(dRMin * dRMin),
      dRMax2(dRMax * dRMax) {}

void PFTkEGAlgo::initRegion(Region &r) const {
  // NOTE: assume imput is sorted already
  r.egphotons.clear();
  r.egeles.clear();
}

void PFTkEGAlgo::runTkEG(Region &r) const {
  initRegion(r);

  if (debug_ > 0) {
    edm::LogInfo("PFTkEGAlgo") << "   # emCalo: " << r.emcalo.size() << " # tk: " << r.track.size();
  }
  if (debug_ > 0) {
    for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
      const auto &calo = r.emcalo[ic];
      edm::LogInfo("PFTkEGAlgo") << "[OLD] IN calo[" << ic << "] pt: " << calo.floatPt()
                                 << " eta: " << r.globalEta(calo.floatEta()) << " phi: " << r.globalPhi(calo.floatPhi())
                                 << " hwEta: " << calo.hwEta << " hwPhi: " << calo.hwPhi
                                 << " src pt: " << calo.src->pt() << " src eta: " << calo.src->eta()
                                 << " src phi: " << calo.src->phi();
    }
  }

  // NOTE: we run this step for all clusters (before matching) as it is done in the pre-PF EG algorithm
  std::vector<int> emCalo2emCalo(r.emcalo.size(), -1);
  if (doBremRecovery_)
    link_emCalo2emCalo(r, emCalo2emCalo);

  // track to EM calo matching
  std::vector<int> emCalo2tk(r.emcalo.size(), -1);
  link_emCalo2tk(r, emCalo2tk);

  // add EG objects to region;
  eg_algo(r, emCalo2emCalo, emCalo2tk);
}

void PFTkEGAlgo::eg_algo(Region &r, const std::vector<int> &emCalo2emCalo, const std::vector<int> &emCalo2tk) const {
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;

    int itk = emCalo2tk[ic];

    // 1. create EG objects before brem recovery
    addEgObjsToPF(r, ic, calo.hwFlags, calo.floatPt(), itk);

    // check if brem recovery is on
    if (!doBremRecovery_)
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
    // FIXME: duplicating the object is suboptimal but this is done for keeping things as in TDR code...
    addEgObjsToPF(r, ic, calo.hwFlags + 1, ptBremReco, itk);
  }
}

EGIsoParticle &PFTkEGAlgo::addEGIsoToPF(std::vector<EGIsoParticle> &egobjs,
                                        const CaloCluster &calo,
                                        const int hwQual,
                                        const float ptCorr) const {
  EGIsoParticle egiso;
  egiso.setFloatPt(ptCorr);
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  egiso.cluster = calo;

  egiso.hwQual = hwQual;

  egiso.hwIso = 0;
  egiso.hwIsoPV = 0;
  egiso.hwPFIso = 0;
  egiso.hwPFIsoPV = 0;
  egiso.ele_idx = -1;
  egobjs.push_back(egiso);
  return egobjs.back();
}

EGIsoEleParticle &PFTkEGAlgo::addEGIsoEleToPF(std::vector<EGIsoEleParticle> &egobjs,
                                              const CaloCluster &calo,
                                              const PropagatedTrack &track,
                                              const int hwQual,
                                              const float ptCorr) const {
  EGIsoEleParticle egiso;
  egiso.setFloatPt(ptCorr);
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  egiso.cluster = calo;

  egiso.hwVtxEta = track.hwVtxEta;
  egiso.hwVtxPhi = track.hwVtxPhi;
  egiso.hwZ0 = track.hwZ0;
  egiso.hwCharge = track.hwCharge;
  egiso.track = track;

  egiso.hwQual = hwQual;
  egiso.hwIso = 0;
  egiso.hwPFIso = 0;
  egobjs.push_back(egiso);
  return egobjs.back();
}

void PFTkEGAlgo::addEgObjsToPF(
    Region &r, const int calo_idx, const int hwQual, const float ptCorr, const int tk_idx) const {
  EGIsoParticle &egobj = addEGIsoToPF(r.egphotons, r.emcalo[calo_idx], hwQual, ptCorr);
  if (tk_idx != -1) {
    egobj.ele_idx = r.egeles.size();
    addEGIsoEleToPF(r.egeles, r.emcalo[calo_idx], r.track[tk_idx], hwQual, ptCorr);
  }
}

void PFTkEGAlgo::link_emCalo2emCalo(const Region &r, std::vector<int> &emCalo2emCalo) const {
  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];
    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;

    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic + 1; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = r.emcalo[jc];
      if (filterHwQuality_ && otherCalo.hwFlags != caloHwQual_)
        continue;

      if (fabs(otherCalo.floatEta() - calo.floatEta()) < dEtaMaxBrem_ &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < dPhiMaxBrem_) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}

void PFTkEGAlgo::link_emCalo2tk(const Region &r, std::vector<int> &emCalo2tk) const {
  for (int ic = 0, nc = r.emcalo.size(); ic < nc; ++ic) {
    auto &calo = r.emcalo[ic];

    if (filterHwQuality_ && calo.hwFlags != caloHwQual_)
      continue;
    // compute elliptic matching
    auto eta_index =
        std::distance(
            absEtaBoundaries_.begin(),
            std::lower_bound(absEtaBoundaries_.begin(), absEtaBoundaries_.end(), r.globalAbsEta(calo.floatEta()))) -
        1;
    float dEtaMax = dEtaValues_[eta_index];
    float dPhiMax = dPhiValues_[eta_index];

    if (debug_ > 4)
      edm::LogInfo("PFTkEGAlgo") << "idx: " << eta_index << " deta: " << dEtaMax << " dphi: " << dPhiMax;

    float dPtMin = 999;
    if (debug_ > 3)
      edm::LogInfo("PFTkEGAlgo") << "--- calo: pt: " << calo.floatPt() << " eta: " << calo.floatEta()
                                 << " phi: " << calo.floatPhi();
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      if (debug_ > 3)
        edm::LogInfo("PFTkEGAlgo") << "  - tk: pt: " << tk.floatPt() << " eta: " << tk.floatEta()
                                   << " phi: " << tk.floatPhi();

      if (tk.floatPt() < trkQualityPtMin_)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared

      if (debug_ > 3)
        edm::LogInfo("PFTkEGAlgo") << " deta: " << fabs(d_eta) << " dphi: " << d_phi << " ell: "
                                   << ((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax));

      if ((((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) < 1.) {
        if (debug_ > 3)
          edm::LogInfo("PFTkEGAlgo") << "    pass elliptic ";
        // NOTE: for now we implement only best pt match. This is NOT what is done in the L1TkElectronTrackProducer
        if (fabs(tk.floatPt() - calo.floatPt()) < dPtMin) {
          if (debug_ > 3)
            edm::LogInfo("PFTkEGAlgo") << "     best pt match: " << fabs(tk.floatPt() - calo.floatPt());
          emCalo2tk[ic] = itk;
          dPtMin = fabs(tk.floatPt() - calo.floatPt());
        }
      }
    }
  }
}

void PFTkEGAlgo::runTkIso(Region &r, const float z0) const {
  compute_isolation_tkEle(r, r.track, tkIsoParametersTkEle_, z0, false);
  compute_isolation_tkEm(r, r.track, tkIsoParametersTkEm_, z0, false);
}

void PFTkEGAlgo::runPFIso(Region &r, const float z0) const {
  compute_isolation_tkEle(r, r.pf, pfIsoParametersTkEle_, z0, true);
  compute_isolation_tkEm(r, r.pf, pfIsoParametersTkEm_, z0, true);
}
