#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegalgo_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <iostream>
#include <bitset>
#include <vector>

using namespace l1ct;

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::PFTkEGAlgoEmuConfig::PFTkEGAlgoEmuConfig(const edm::ParameterSet &pset)
    : PFTkEGAlgoEmuConfig(pset.getParameter<uint32_t>("nTRACK"),
                          pset.getParameter<uint32_t>("nTRACK_EGIN"),
                          pset.getParameter<uint32_t>("nEMCALO_EGIN"),
                          pset.getParameter<uint32_t>("nEM_EGOUT"),
                          pset.getParameter<bool>("filterHwQuality"),
                          pset.getParameter<bool>("doBremRecovery"),
                          pset.getParameter<bool>("writeBeforeBremRecovery"),
                          pset.getParameter<int>("caloHwQual"),
                          pset.getParameter<bool>("doEndcapHwQual"),
                          pset.getParameter<double>("caloEtMin"),
                          pset.getParameter<double>("dEtaMaxBrem"),
                          pset.getParameter<double>("dPhiMaxBrem"),
                          pset.getParameter<std::vector<double>>("absEtaBoundaries"),
                          pset.getParameter<std::vector<double>>("dEtaValues"),
                          pset.getParameter<std::vector<double>>("dPhiValues"),
                          pset.getParameter<double>("trkQualityPtMin"),
                          pset.getParameter<uint32_t>("algorithm"),
                          pset.getParameter<uint32_t>("nCompCandPerCluster"),
                          pset.getParameter<bool>("writeEGSta"),
                          IsoParameters(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEle")),
                          IsoParameters(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEm")),
                          IsoParameters(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEle")),
                          IsoParameters(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEm")),
                          pset.getParameter<bool>("doTkIso"),
                          pset.getParameter<bool>("doPfIso"),
                          static_cast<EGIsoEleObjEmu::IsoType>(pset.getParameter<uint32_t>("hwIsoTypeTkEle")),
                          static_cast<EGIsoObjEmu::IsoType>(pset.getParameter<uint32_t>("hwIsoTypeTkEm")),
                          pset.getParameter<edm::ParameterSet>("compositeParametersTkEle"),
                          pset.getUntrackedParameter<uint32_t>("debug", 0)) {}

edm::ParameterSetDescription l1ct::PFTkEGAlgoEmuConfig::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<unsigned int>("nTRACK");
  description.add<unsigned int>("nTRACK_EGIN");
  description.add<unsigned int>("nEMCALO_EGIN");
  description.add<unsigned int>("nEM_EGOUT");
  description.add<bool>("doBremRecovery", false);
  description.add<bool>("writeBeforeBremRecovery", false);
  description.add<bool>("filterHwQuality", false);
  description.add<int>("caloHwQual", 4);
  description.add<bool>("doEndcapHwQual", false);
  description.add<double>("dEtaMaxBrem", 0.02);
  description.add<double>("dPhiMaxBrem", 0.1);
  description.add<std::vector<double>>("absEtaBoundaries",
                                       {
                                           0.0,
                                           0.9,
                                           1.5,
                                       });
  description.add<std::vector<double>>("dEtaValues",
                                       {
                                           0.025,
                                           0.015,
                                           0.01,
                                       });
  description.add<std::vector<double>>("dPhiValues",
                                       {
                                           0.07,
                                           0.07,
                                           0.07,
                                       });
  description.add<double>("caloEtMin", 0.0);
  description.add<double>("trkQualityPtMin", 10.0);
  description.add<bool>("writeEGSta", false);
  description.add<edm::ParameterSetDescription>("tkIsoParametersTkEm", IsoParameters::getParameterSetDescription());
  description.add<edm::ParameterSetDescription>("tkIsoParametersTkEle", IsoParameters::getParameterSetDescription());
  description.add<edm::ParameterSetDescription>("pfIsoParametersTkEm", IsoParameters::getParameterSetDescription());
  description.add<edm::ParameterSetDescription>("pfIsoParametersTkEle", IsoParameters::getParameterSetDescription());
  description.add<bool>("doTkIso", true);
  description.add<bool>("doPfIso", true);
  description.add<unsigned int>("hwIsoTypeTkEle", 0);
  description.add<unsigned int>("hwIsoTypeTkEm", 2);
  description.add<unsigned int>("algorithm", 0);
  description.add<unsigned int>("nCompCandPerCluster", 3);
  description.add<edm::ParameterSetDescription>("compositeParametersTkEle",
                                                CompIDParameters::getParameterSetDescription());
  return description;
}

l1ct::PFTkEGAlgoEmuConfig::IsoParameters::IsoParameters(const edm::ParameterSet &pset)
    : IsoParameters(pset.getParameter<double>("tkQualityPtMin"),
                    pset.getParameter<double>("dZ"),
                    pset.getParameter<double>("dRMin"),
                    pset.getParameter<double>("dRMax")) {}

edm::ParameterSetDescription l1ct::PFTkEGAlgoEmuConfig::IsoParameters::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<double>("tkQualityPtMin");
  description.add<double>("dZ", 0.6);
  description.add<double>("dRMin");
  description.add<double>("dRMax");
  return description;
}

l1ct::PFTkEGAlgoEmuConfig::CompIDParameters::CompIDParameters(const edm::ParameterSet &pset)
    : CompIDParameters(pset.getParameter<double>("loose_wp"),
                       pset.getParameter<double>("tight_wp"),
                       pset.getParameter<std::string>("model")) {}

edm::ParameterSetDescription l1ct::PFTkEGAlgoEmuConfig::CompIDParameters::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<double>("loose_wp", -0.732422);
  description.add<double>("tight_wp", 0.214844);
  description.add<std::string>("model", "L1Trigger/Phase2L1ParticleFlow/data/compositeID.json");
  return description;
}
#endif

PFTkEGAlgoEmulator::PFTkEGAlgoEmulator(const PFTkEGAlgoEmuConfig &config)
    : cfg(config), model_(nullptr), debug_(cfg.debug) {
  if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v0 ||
      cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEB_v0 ||
      cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v1) {
#ifdef CMSSW_GIT_HASH
    auto resolvedFileName = edm::FileInPath(cfg.compIDparams.conifer_model).fullPath();
#else
    auto resolvedFileName = cfg.compIDparams.conifer_model;
#endif
    if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v0) {
      model_ = new conifer::BDT<bdt_feature_t, bdt_score_t, false>(resolvedFileName);
    } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEB_v0) {
      model_ = new conifer::BDT<bdt_eb_feature_t, bdt_eb_score_t, false>(resolvedFileName);
    } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v1) {
      model_ = new conifer::BDT<bdt_ee_feature_t, bdt_ee_score_t, false>(resolvedFileName);
    }
  }
}

void PFTkEGAlgoEmulator::toFirmware(const PFInputRegion &in,
                                    PFRegion &region,
                                    EmCaloObj emcalo[/*nCALO*/],
                                    TkObj track[/*nTRACK*/]) const {
  region = in.region;
  l1ct::toFirmware(in.track, cfg.nTRACK_EGIN, track);
  l1ct::toFirmware(in.emcalo, cfg.nEMCALO_EGIN, emcalo);
  if (debug_ > 0)
    dbgCout() << "# of inpput tracks: " << in.track.size() << " (max: " << cfg.nTRACK_EGIN << ")"
              << " emcalo: " << in.emcalo.size() << "(" << cfg.nEMCALO_EGIN << ")" << std::endl;
}

void PFTkEGAlgoEmulator::toFirmware(const OutputRegion &out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const {
  l1ct::toFirmware(out.egphoton, cfg.nEM_EGOUT, out_egphs);
  l1ct::toFirmware(out.egelectron, cfg.nEM_EGOUT, out_egeles);
  if (debug_ > 0)
    dbgCout() << "# output photons: " << out.egphoton.size() << " electrons: " << out.egelectron.size() << std::endl;
}

void PFTkEGAlgoEmulator::toFirmware(
    const PFInputRegion &in, const l1ct::PVObjEmu &pvin, PFRegion &region, TkObj track[/*nTRACK*/], PVObj &pv) const {
  region = in.region;
  l1ct::toFirmware(in.track, cfg.nTRACK, track);
  pv = pvin;
  if (debug_ > 0)
    dbgCout() << "# of inpput tracks: " << in.track.size() << " (max: " << cfg.nTRACK << ")" << std::endl;
}

float PFTkEGAlgoEmulator::deltaPhi(float phi1, float phi2) const {
  // reduce to [-pi,pi]
  float x = phi1 - phi2;
  float o2pi = 1. / (2. * M_PI);
  if (std::abs(x) <= float(M_PI))
    return x;
  float n = std::round(x * o2pi);
  return x - n * float(2. * M_PI);
}

void PFTkEGAlgoEmulator::link_emCalo2emCalo(const std::vector<EmCaloObjEmu> &emcalo,
                                            std::vector<int> &emCalo2emCalo) const {
  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];
    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic + 1; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = emcalo[jc];

      if (fabs(otherCalo.floatEta() - calo.floatEta()) < cfg.dEtaMaxBrem &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < cfg.dPhiMaxBrem) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}

void PFTkEGAlgoEmulator::link_emCalo2tk_elliptic(const PFRegionEmu &r,
                                                 const std::vector<EmCaloObjEmu> &emcalo,
                                                 const std::vector<TkObjEmu> &track,
                                                 std::vector<int> &emCalo2tk) const {
  unsigned int nTrackMax = std::min<unsigned>(track.size(), cfg.nTRACK_EGIN);
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    float dPtMin = 999;
    for (unsigned int itk = 0; itk < nTrackMax; ++itk) {
      const auto &tk = track[itk];
      if (tk.floatPt() < cfg.trkQualityPtMin)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared

      auto eta_index =
          std::distance(cfg.absEtaBoundaries.begin(),
                        std::lower_bound(
                            cfg.absEtaBoundaries.begin(), cfg.absEtaBoundaries.end(), abs(r.floatGlbEta(calo.hwEta)))) -
          1;

      float dEtaMax = cfg.dEtaValues[eta_index];
      float dPhiMax = cfg.dPhiValues[eta_index];

      if (debug_ > 2 && calo.hwPt > 0) {
        dbgCout() << "[REF] tried to link calo " << ic << " (pt " << calo.intPt() << ", eta " << calo.intEta()
                  << ", phi " << calo.intPhi() << ") "
                  << " to tk " << itk << " (pt " << tk.intPt() << ", eta " << tk.intEta() << ", phi " << tk.intPhi()
                  << "): "
                  << " eta_index " << eta_index << ", "
                  << " dEta " << d_eta << " (max " << dEtaMax << "), dPhi " << d_phi << " (max " << dPhiMax << ") "
                  << " ellipse = "
                  << (((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) << "\n";
      }
      if ((((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) < 1.) {
        // NOTE: for now we implement only best pt match. This is NOT what is done in the L1TkElectronTrackProducer
        if (fabs(tk.floatPt() - calo.floatPt()) < dPtMin) {
          emCalo2tk[ic] = itk;
          dPtMin = fabs(tk.floatPt() - calo.floatPt());
        }
      }
    }
  }
}

void PFTkEGAlgoEmulator::link_emCalo2tk_composite_eb_ee(const PFRegionEmu &r,
                                                        const std::vector<EmCaloObjEmu> &emcalo,
                                                        const std::vector<TkObjEmu> &track,
                                                        std::vector<int> &emCalo2tk,
                                                        std::vector<id_score_t> &emCaloTkBdtScore) const {
  unsigned int nTrackMax = std::min<unsigned>(track.size(), cfg.nTRACK_EGIN);
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    std::vector<CompositeCandidate> candidates;
    float sumTkPt = 0.;
    unsigned int nTkMatch = 0;
    for (unsigned int itk = 0; itk < nTrackMax; ++itk) {
      const auto &tk = track[itk];
      if (tk.floatPt() <= cfg.trkQualityPtMin)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared
      // float dR2 = (d_phi * d_phi) + (d_eta * d_eta);
      const float dPhiMax = 0.3;  // FIXME: configure
      const float dEtaMax = 0.03;

      bool keep = false;
      if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v0) {
        keep = (d_phi * d_phi) + (d_eta * d_eta) < 0.04;
      } else if ((cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEB_v0) ||
                 (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v1)) {
        keep = (((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) < 1.;
      }
      if (keep) {
        // Only store indices, dR and dpT for now. The other quantities are computed only for the best nCandPerCluster.
        CompositeCandidate cand;
        cand.cluster_idx = ic;
        cand.track_idx = itk;
        cand.dpt = std::abs(tk.floatPt() - calo.floatPt());
        candidates.push_back(cand);
        sumTkPt += tk.floatPt();
        nTkMatch++;
      }
    }
    // we use dpt as sort criteria
    std::sort(candidates.begin(),
              candidates.end(),
              [](const CompositeCandidate &a, const CompositeCandidate &b) -> bool { return a.dpt < b.dpt; });
    unsigned int nCandPerCluster = std::min<unsigned int>(candidates.size(), cfg.nCompCandPerCluster);
    if (nCandPerCluster == 0)
      continue;

    id_score_t maxScore = -(1 << (l1ct::id_score_t::iwidth - 1));
    int ibest = -1;
    for (unsigned int icand = 0; icand < nCandPerCluster; icand++) {
      auto &cand = candidates[icand];
      const std::vector<EmCaloObjEmu> &emcalo_sel = emcalo;
      id_score_t score = 0;
      if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v0) {
        score = compute_composite_score(cand, emcalo_sel, track, cfg.compIDparams);
      } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEB_v0) {
        score = compute_composite_score_eb(cand, sumTkPt, nTkMatch, emcalo_sel, track, cfg.compIDparams);
      } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v1) {
        score = compute_composite_score_ee(cand, sumTkPt, nTkMatch, emcalo_sel, track, cfg.compIDparams);
      }
      if ((score > cfg.compIDparams.bdtScore_loose_wp) && (score > maxScore)) {
        maxScore = score;
        ibest = icand;
      }
    }
    if (ibest != -1) {
      emCalo2tk[ic] = candidates[ibest].track_idx;
      emCaloTkBdtScore[ic] = maxScore;
    }
  }
}

id_score_t PFTkEGAlgoEmulator::compute_composite_score_eb(CompositeCandidate &cand,
                                                          float sumTkPt,
                                                          unsigned int nTkMatch,
                                                          const std::vector<EmCaloObjEmu> &emcalo,
                                                          const std::vector<TkObjEmu> &track,
                                                          const PFTkEGAlgoEmuConfig::CompIDParameters &params) const {
  // Get the cluster/track objects that form the composite candidate
  const auto &calo = emcalo[cand.cluster_idx];
  const auto &tk = track[cand.track_idx];
  const l1tp2::CaloCrystalCluster *crycl = dynamic_cast<const l1tp2::CaloCrystalCluster *>(calo.src);

  // Prepare the input features
  // FIXME: use the EmCaloObj and TkObj to get all the features
  // FIXME: make sure that all input features end up in the PFCluster and PFTrack objects with the right precision

  // FIXME: 16 bit estimate for the inversion is approximate
  ap_ufixed<16, 0> calo_invPt = l1ct::invert_with_shift<pt_t, ap_ufixed<16, 0>, 1024>(calo.hwPt);
  // NOTE: this could be computed once per cluster and passed directly to the function
  ap_ufixed<16, 0> sumTk_invPt = l1ct::invert_with_shift<pt_t, ap_ufixed<16, 0>, 1024>(pt_t(sumTkPt));
  ap_ufixed<16, 0> tk_invPt = l1ct::invert_with_shift<pt_t, ap_ufixed<16, 0>, 1024>(tk.hwPt);

  constexpr std::array<float, 1 << l1ct::redChi2Bin_t::width> chi2RPhiBins = {
      {0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0}};

  bdt_eb_feature_t cl_pt = calo.floatPt();
  bdt_eb_feature_t cl_ss = crycl->e2x5() / crycl->e5x5();
  bdt_eb_feature_t cl_relIso = iso_t(crycl->isolation()) * calo_invPt;
  bdt_eb_feature_t cl_staWP = calo.hwEmID & 0x1;
  bdt_eb_feature_t cl_looseTkWP = calo.hwEmID & 0x2;
  bdt_eb_feature_t tk_chi2RPhi = chi2RPhiBins[tk.hwRedChi2RPhi.to_int()];  // FIXME: should switch to bin #
  bdt_eb_feature_t tk_ptFrac = tk.hwPt * sumTk_invPt;                      // FIXME: could this become sum_tk/calo pt?
  bdt_eb_feature_t cltk_ptRatio =
      calo.hwPt * tk_invPt;  // FIXME: could we use the inverse so that we compute the inverse of caloPt once?
  bdt_eb_feature_t cltk_nTkMatch = nTkMatch;
  bdt_eb_feature_t cltk_absDeta = fabs(tk.floatEta() - calo.floatEta());  // FIXME: switch to hwEta diff
  bdt_eb_feature_t cltk_absDphi = fabs(tk.floatPhi() - calo.floatPhi());  // FIXME: switch to hwPhi diff

  // Run BDT inference
  std::vector<bdt_eb_feature_t> inputs = {cl_pt,
                                          cl_ss,
                                          cl_relIso,
                                          cl_staWP,
                                          cl_looseTkWP,
                                          tk_chi2RPhi,
                                          tk_ptFrac,
                                          cltk_ptRatio,
                                          cltk_nTkMatch,
                                          cltk_absDeta,
                                          cltk_absDphi};
  auto *composite_bdt_eb_ = static_cast<conifer::BDT<bdt_eb_feature_t, bdt_eb_score_t, false> *>(model_);
  std::vector<bdt_eb_score_t> bdt_score = composite_bdt_eb_->decision_function(inputs);
  // std::cout << "  out BDT score: " << bdt_score[0] << std::endl;
  constexpr unsigned int MAX_SCORE = 1 << (bdt_eb_score_t::iwidth - 1);
  return bdt_score[0] / bdt_eb_score_t(MAX_SCORE);  // normalize to [-1,1]
}

id_score_t PFTkEGAlgoEmulator::compute_composite_score_ee(CompositeCandidate &cand,
                                                          float sumTkPt,
                                                          unsigned int nTkMatch,
                                                          const std::vector<EmCaloObjEmu> &emcalo,
                                                          const std::vector<TkObjEmu> &track,
                                                          const PFTkEGAlgoEmuConfig::CompIDParameters &params) const {
  // Get the cluster/track objects that form the composite candidate
  const auto &calo = emcalo[cand.cluster_idx];
  const auto &tk = track[cand.track_idx];
  const l1t::PFTrack *pftk = tk.src;
  const l1t::HGCalMulticluster *cl3d = dynamic_cast<const l1t::HGCalMulticluster *>(calo.src);

  // Prepare the input features
  bdt_ee_feature_t cl_coreshowerlength = cl3d->coreShowerLength();
  bdt_ee_feature_t cl_meanz = std::fabs(cl3d->zBarycenter());
  bdt_ee_feature_t cl_spptot = cl3d->sigmaPhiPhiTot();
  bdt_ee_feature_t cl_seetot = cl3d->sigmaEtaEtaTot();
  bdt_ee_feature_t cl_szz = cl3d->sigmaZZ();
  bdt_ee_feature_t cl_multiClassPionIdScore = calo.floatPiProb();
  bdt_ee_feature_t cl_multiClassEmIdScore = calo.floatEmProb();
  bdt_ee_feature_t tk_ptFrac = pftk->pt() / sumTkPt;
  bdt_ee_feature_t cltk_ptRatio = calo.floatPt() / pftk->pt();
  bdt_ee_feature_t cltk_absDeta = fabs(cl3d->eta() - pftk->caloEta());
  bdt_ee_feature_t cltk_absDphi = fabs(cl3d->phi() - pftk->caloPhi());

  // Run BDT inference
  std::vector<bdt_ee_feature_t> inputs = {cl_coreshowerlength,
                                          cl_meanz,
                                          cl_spptot,
                                          cl_seetot,
                                          cl_szz,
                                          cl_multiClassPionIdScore,
                                          cl_multiClassEmIdScore,
                                          tk_ptFrac,
                                          cltk_ptRatio,
                                          cltk_absDeta,
                                          cltk_absDphi};
  auto *composite_bdt_ee_ = static_cast<conifer::BDT<bdt_ee_feature_t, bdt_ee_score_t, false> *>(model_);
  std::vector<bdt_ee_score_t> bdt_score = composite_bdt_ee_->decision_function(inputs);
  // std::cout << "  out BDT score: " << bdt_score[0] << std::endl;
  constexpr unsigned int MAX_SCORE = 1 << (bdt_ee_score_t::iwidth - 1);
  return bdt_score[0] / bdt_ee_score_t(MAX_SCORE);
}

id_score_t PFTkEGAlgoEmulator::compute_composite_score(CompositeCandidate &cand,
                                                       const std::vector<EmCaloObjEmu> &emcalo,
                                                       const std::vector<TkObjEmu> &track,
                                                       const PFTkEGAlgoEmuConfig::CompIDParameters &params) const {
  // Get the cluster/track objects that form the composite candidate
  const auto &calo = emcalo[cand.cluster_idx];
  const auto &tk = track[cand.track_idx];

  // Prepare the input features
  bdt_feature_t hoe = calo.hwHoe;
  bdt_feature_t tkpt = tk.hwPt;
  bdt_feature_t srrtot = calo.hwSrrTot;
  bdt_feature_t deta = tk.hwEta - calo.hwEta;
  ap_ufixed<18, 0> calo_invPt = l1ct::invert_with_shift<pt_t, ap_ufixed<18, 0>, 1024>(calo.hwPt);
  bdt_feature_t dpt = tk.hwPt * calo_invPt;
  bdt_feature_t meanz = calo.hwMeanZ;
  bdt_feature_t dphi = tk.hwPhi - calo.hwPhi;
  bdt_feature_t nstubs = tk.hwStubs;
  bdt_feature_t chi2rphi = tk.hwRedChi2RPhi;
  bdt_feature_t chi2rz = tk.hwRedChi2RZ;
  bdt_feature_t chi2bend = tk.hwRedChi2Bend;

  // Run BDT inference
  std::vector<bdt_feature_t> inputs = {tkpt, hoe, srrtot, deta, dphi, dpt, meanz, nstubs, chi2rphi, chi2rz, chi2bend};
  auto *composite_bdt_ = static_cast<conifer::BDT<bdt_feature_t, bdt_score_t, false> *>(model_);
  std::vector<bdt_score_t> bdt_score = composite_bdt_->decision_function(inputs);
  constexpr unsigned int MAX_SCORE = (1 << (bdt_score_t::iwidth - 1));
  return bdt_score[0] / bdt_score_t(MAX_SCORE);
}

void PFTkEGAlgoEmulator::sel_emCalo(unsigned int nmax_sel,
                                    const std::vector<EmCaloObjEmu> &emcalo,
                                    std::vector<EmCaloObjEmu> &emcalo_sel) const {
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    const auto &calo = emcalo[ic];
    if ((calo.hwPt == 0) || (cfg.filterHwQuality && calo.hwEmID != cfg.caloHwQual) ||
        (calo.floatPt() < cfg.emClusterPtMin))
      continue;
    emcalo_sel.push_back(calo);
    if (emcalo_sel.size() >= nmax_sel)
      break;
  }
}

void PFTkEGAlgoEmulator::run(const PFInputRegion &in, OutputRegion &out) const {
  if (debug_ > 1) {
    for (int ic = 0, nc = in.emcalo.size(); ic < nc; ++ic) {
      const auto &calo = in.emcalo[ic];
      if (calo.hwPt > 0)
        dbgCout() << "[REF] IN calo[" << ic << "] pt: " << calo.hwPt << " eta: " << calo.hwEta
                  << " (glb eta: " << in.region.floatGlbEta(calo.hwEta) << ") phi: " << calo.hwPhi
                  << "(glb phi: " << in.region.floatGlbPhi(calo.hwPhi) << ") qual: " << std::bitset<4>(calo.hwEmID)
                  << std::endl;
    }
  }
  // FIXME: can be removed in the endcap since now running with the "interceptor".
  // Might still be needed in barrel
  // filter and select first N elements of input clusters
  std::vector<EmCaloObjEmu> emcalo_sel;
  sel_emCalo(cfg.nEMCALO_EGIN, in.emcalo, emcalo_sel);

  std::vector<int> emCalo2emCalo(emcalo_sel.size(), -1);
  if (cfg.doBremRecovery)
    link_emCalo2emCalo(emcalo_sel, emCalo2emCalo);

  std::vector<int> emCalo2tk(emcalo_sel.size(), -1);
  std::vector<id_score_t> emCaloTkBdtScore(emcalo_sel.size(), 0);

  if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::elliptic) {
    link_emCalo2tk_elliptic(in.region, emcalo_sel, in.track, emCalo2tk);
  } else {
    link_emCalo2tk_composite_eb_ee(in.region, emcalo_sel, in.track, emCalo2tk, emCaloTkBdtScore);
  }

  out.egsta.clear();
  std::vector<EGIsoObjEmu> egobjs;
  std::vector<EGIsoEleObjEmu> egeleobjs;
  eg_algo(in.region, emcalo_sel, in.track, emCalo2emCalo, emCalo2tk, emCaloTkBdtScore, out.egsta, egobjs, egeleobjs);

  unsigned int nEGOut = std::min<unsigned>(cfg.nEM_EGOUT, egobjs.size());
  unsigned int nEGEleOut = std::min<unsigned>(cfg.nEM_EGOUT, egeleobjs.size());

  // init output containers
  out.egphoton.clear();
  out.egelectron.clear();
  ptsort_ref(egobjs.size(), nEGOut, egobjs, out.egphoton);
  ptsort_ref(egeleobjs.size(), nEGEleOut, egeleobjs, out.egelectron);
}

void PFTkEGAlgoEmulator::eg_algo(const PFRegionEmu &region,
                                 const std::vector<EmCaloObjEmu> &emcalo,
                                 const std::vector<TkObjEmu> &track,
                                 const std::vector<int> &emCalo2emCalo,
                                 const std::vector<int> &emCalo2tk,
                                 const std::vector<id_score_t> &emCaloTkBdtScore,
                                 std::vector<EGObjEmu> &egstas,
                                 std::vector<EGIsoObjEmu> &egobjs,
                                 std::vector<EGIsoEleObjEmu> &egeleobjs) const {
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    // discard immediately EG objects that would not fall in the fiducial eta-phi region
    if (!region.isFiducial(calo))
      continue;

    if (debug_ > 3)
      dbgCout() << "[REF] SEL emcalo with pt: " << calo.hwPt << " qual: " << calo.hwEmID << " eta: " << calo.hwEta
                << " phi " << calo.hwPhi << std::endl;

    int itk = emCalo2tk[ic];
    const id_score_t &bdt = emCaloTkBdtScore[ic];

    // check if brem recovery is on
    if (!cfg.doBremRecovery || cfg.writeBeforeBremRecovery) {
      // 1. create EG objects before brem recovery
      unsigned int egQual = calo.hwEmID;
      // If we write both objects with and without brem-recovery
      // bit 3 is used for the brem-recovery bit: if set = no recovery
      // (for consistency with the barrel hwQual where by default the brem recovery is done upstream)
      if (cfg.writeBeforeBremRecovery && cfg.doBremRecovery) {
        egQual = calo.hwEmID | 0x8;
      }

      addEgObjsToPF(egstas, egobjs, egeleobjs, emcalo, track, ic, egQual, calo.hwPt, itk, bdt);
    }

    if (!cfg.doBremRecovery)
      continue;

    // check if the cluster has already been used in a brem reclustering
    if (emCalo2emCalo[ic] != -1)
      continue;

    pt_t ptBremReco = calo.hwPt;
    std::vector<unsigned int> components;

    for (int jc = ic; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] == ic) {
        auto &otherCalo = emcalo[jc];
        ptBremReco += otherCalo.hwPt;
        components.push_back(jc);
      }
    }

    // 2. create EG objects with brem recovery
    addEgObjsToPF(egstas, egobjs, egeleobjs, emcalo, track, ic, calo.hwEmID, ptBremReco, itk, bdt, components);
  }
}

EGObjEmu &PFTkEGAlgoEmulator::addEGStaToPF(std::vector<EGObjEmu> &egobjs,
                                           const EmCaloObjEmu &calo,
                                           const unsigned int hwQual,
                                           const pt_t ptCorr,
                                           const std::vector<unsigned int> &components) const {
  EGObjEmu egsta;
  egsta.clear();
  egsta.hwPt = ptCorr;
  egsta.hwEta = calo.hwEta;
  egsta.hwPhi = calo.hwPhi;
  egsta.hwQual = hwQual;
  egobjs.push_back(egsta);

  if (debug_ > 2)
    dbgCout() << "[REF] EGSta pt: " << egsta.hwPt << " eta: " << egsta.hwEta << " phi: " << egsta.hwPhi
              << " qual: " << std::bitset<4>(egsta.hwQual) << " packed: " << egsta.pack().to_string(16) << std::endl;

  return egobjs.back();
}

EGIsoObjEmu &PFTkEGAlgoEmulator::addEGIsoToPF(std::vector<EGIsoObjEmu> &egobjs,
                                              const EmCaloObjEmu &calo,
                                              const unsigned int hwQual,
                                              const pt_t ptCorr) const {
  EGIsoObjEmu egiso;
  egiso.clear();
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  unsigned int egHwQual = hwQual;
  if (cfg.doEndcapHwQual) {
    // 1. zero-suppress the loose EG-ID (bit 1)
    // 2. for now use the standalone tight definition (bit 0) to set the tight point for photons (bit 2)
    egHwQual = (hwQual & 0x9) | ((hwQual & 0x1) << 2);
  }
  egiso.hwQual = egHwQual;
  egiso.srcCluster = calo.src;
  egobjs.push_back(egiso);

  if (debug_ > 2)
    dbgCout() << "[REF] EGIsoObjEmu pt: " << egiso.hwPt << " eta: " << egiso.hwEta << " phi: " << egiso.hwPhi
              << " qual: " << std::bitset<4>(egiso.hwQual) << " packed: " << egiso.pack().to_string(16) << std::endl;

  return egobjs.back();
}

EGIsoEleObjEmu &PFTkEGAlgoEmulator::addEGIsoEleToPF(std::vector<EGIsoEleObjEmu> &egobjs,
                                                    const EmCaloObjEmu &calo,
                                                    const TkObjEmu &track,
                                                    const unsigned int hwQual,
                                                    const pt_t ptCorr,
                                                    const id_score_t bdtScore) const {
  EGIsoEleObjEmu egiso;
  egiso.clear();
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  unsigned int egHwQual = hwQual;
  if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v0) {
    // tight ele WP is set for tight BDT score
    egHwQual = (hwQual & 0x9) | ((bdtScore >= cfg.compIDparams.bdtScore_tight_wp) << 1);
  } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEB_v0) {
    vector<float> pt_bins = {0, 5, 10, 20, 30, 50};
    vector<float> tight_wps = {0.19, 0.05, -0.35, -0.45, -0.5, -0.38};

    bool isTight = false;

    // std::upper_bound returns an iterator to the first element in pt_bins that is greater than pt_value
    float pt_value = egiso.floatPt();
    auto it = std::upper_bound(pt_bins.begin(), pt_bins.end(), pt_value);
    unsigned int bin_index = it - pt_bins.begin() - 1;

    isTight = (bdtScore > id_score_t(tight_wps[bin_index]));

    // tight ele WP is set for tight BDT score
    egHwQual = (hwQual & 0x9) | (isTight << 1);
  } else if (cfg.algorithm == PFTkEGAlgoEmuConfig::Algo::compositeEE_v1) {
    vector<float> pt_bins = {0, 5, 10, 20, 30, 50};
    vector<float> tight_wps = {
        1.2367626271489272,
        0.3639653772014115,
        -0.8472978603872036,
        -0.8953840470548413,
        -0.7537718023763801,
        -0.6190392084062235,
    };

    bool isTight = false;
    // std::upper_bound returns an iterator to the first element in pt_bins that is greater than pt_value
    float pt_value = egiso.floatPt();
    auto it = std::upper_bound(pt_bins.begin(), pt_bins.end(), pt_value);
    unsigned int bin_index = it - pt_bins.begin() - 1;

    isTight = (bdtScore > id_score_t(tight_wps[bin_index] / 4.));

    // tight ele WP is set for tight BDT score
    egHwQual = (hwQual & 0x9) | (isTight << 1);
  } else {
    if (cfg.doEndcapHwQual) {
      // 1. zero-suppress the loose EG-ID (bit 1)
      // 2. for now use the standalone tight definition (bit 0) to set the tight point for eles (bit 1)
      egHwQual = (hwQual & 0x9) | ((hwQual & 0x1) << 1);
    }
  }
  egiso.hwQual = egHwQual;
  egiso.hwDEta = track.hwVtxEta() - egiso.hwEta;
  egiso.hwDPhi = abs(track.hwVtxPhi() - egiso.hwPhi);
  egiso.hwZ0 = track.hwZ0;
  egiso.hwCharge = track.hwCharge;
  egiso.srcCluster = calo.src;
  egiso.srcTrack = track.src;
  egiso.hwIDScore = bdtScore;
  egobjs.push_back(egiso);

  if (debug_ > 2)
    dbgCout() << "[REF] EGIsoEleObjEmu pt: " << egiso.hwPt << " eta: " << egiso.hwEta << " phi: " << egiso.hwPhi
              << " qual: " << std::bitset<4>(egiso.hwQual) << " packed: " << egiso.pack().to_string(16) << std::endl;

  return egobjs.back();
}

void PFTkEGAlgoEmulator::addEgObjsToPF(std::vector<EGObjEmu> &egstas,
                                       std::vector<EGIsoObjEmu> &egobjs,
                                       std::vector<EGIsoEleObjEmu> &egeleobjs,
                                       const std::vector<EmCaloObjEmu> &emcalo,
                                       const std::vector<TkObjEmu> &track,
                                       const int calo_idx,
                                       const unsigned int hwQual,
                                       const pt_t ptCorr,
                                       const int tk_idx,
                                       const id_score_t bdtScore,
                                       const std::vector<unsigned int> &components) const {
  int src_idx = -1;
  if (writeEgSta()) {
    addEGStaToPF(egstas, emcalo[calo_idx], hwQual, ptCorr, components);
    src_idx = egstas.size() - 1;
  }
  EGIsoObjEmu &egobj = addEGIsoToPF(egobjs, emcalo[calo_idx], hwQual, ptCorr);
  egobj.src_idx = src_idx;
  if (tk_idx != -1) {
    EGIsoEleObjEmu &eleobj = addEGIsoEleToPF(egeleobjs, emcalo[calo_idx], track[tk_idx], hwQual, ptCorr, bdtScore);
    eleobj.src_idx = src_idx;
  }
}

void PFTkEGAlgoEmulator::runIso(const PFInputRegion &in,
                                const std::vector<l1ct::PVObjEmu> &pvs,
                                OutputRegion &out) const {
  if (cfg.doTkIso) {
    compute_isolation(out.egelectron, in.track, cfg.tkIsoParams_tkEle, pvs[0].hwZ0);
    compute_isolation(out.egphoton, in.track, cfg.tkIsoParams_tkEm, pvs[0].hwZ0);
  }
  if (cfg.doPfIso) {
    compute_isolation(out.egelectron, out.pfcharged, out.pfneutral, cfg.pfIsoParams_tkEle, pvs[0].hwZ0);
    compute_isolation(out.egphoton, out.pfcharged, out.pfneutral, cfg.pfIsoParams_tkEm, pvs[0].hwZ0);
  }

  std::for_each(out.egelectron.begin(), out.egelectron.end(), [&](EGIsoEleObjEmu &obj) {
    obj.hwIso = obj.hwIsoVar(cfg.hwIsoTypeTkEle);
  });
  std::for_each(
      out.egphoton.begin(), out.egphoton.end(), [&](EGIsoObjEmu &obj) { obj.hwIso = obj.hwIsoVar(cfg.hwIsoTypeTkEm); });
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                                           const std::vector<TkObjEmu> &objects,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egphoton = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, objects, cfg.nTRACK, egphoton, params, z0);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::TkIso, sumPt);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::TkIsoPV, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                                           const std::vector<TkObjEmu> &objects,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egele = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, objects, cfg.nTRACK, egele, params, z0);
    egele.setHwIso(EGIsoEleObjEmu::IsoType::TkIso, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                                           const std::vector<PFChargedObjEmu> &charged,
                                           const std::vector<PFNeutralObjEmu> &neutrals,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egphoton = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    // FIXME: set max # of PF objects for iso
    compute_sumPt(sumPt, sumPtPV, charged, charged.size(), egphoton, params, z0);
    compute_sumPt(sumPt, sumPtPV, neutrals, neutrals.size(), egphoton, params, z0);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::PfIso, sumPt);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::PfIsoPV, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                                           const std::vector<PFChargedObjEmu> &charged,
                                           const std::vector<PFNeutralObjEmu> &neutrals,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egele = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, charged, charged.size(), egele, params, z0);
    compute_sumPt(sumPt, sumPtPV, neutrals, neutrals.size(), egele, params, z0);
    egele.setHwIso(EGIsoEleObjEmu::IsoType::PfIso, sumPtPV);
  }
}
