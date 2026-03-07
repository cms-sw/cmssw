#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/hgcalinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/nn_activation.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::HgcalClusterDecoderEmulator::HgcalClusterDecoderEmulator(const edm::ParameterSet &pset)
    : slim_(pset.getParameter<bool>("slim")),
      multiclass_id_(pset.getParameterSet("multiclass_id")),
      corrector_(pset.getParameter<std::string>("corrector"),
                 pset.getParameter<double>("correctorEmfMax"),
                 false,
                 pset.getParameter<bool>("emulateCorrections"),
                 l1tpf::corrector::EmulationMode::Correction),
      emInterpScenario_(setEmInterpScenario(pset.getParameter<std::string>("emInterpScenario"))) {}

edm::ParameterSetDescription l1ct::HgcalClusterDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<bool>("slim", false);
  description.add<std::string>("corrector", "");
  description.add<double>("correctorEmfMax", -1);
  description.add<bool>("emulateCorrections", false);
  description.add<edm::ParameterSetDescription>("multiclass_id", MultiClassID::getParameterSetDescription());
  description.add<std::string>("emInterpScenario", "No");
  return description;
}

l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs::WPs(const edm::ParameterSet &pset)
    : l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs::WPs(
          pset.getParameter<std::vector<double>>("wp_pt"),
          pset.getParameter<std::vector<double>>("wp_PU"),
          pset.getParameter<std::vector<double>>("wp_Pi"),
          pset.getParameter<std::vector<double>>("wp_PFEm"),
          pset.getParameter<std::vector<double>>("wp_EgEm"),
          pset.getParameter<std::vector<double>>("wp_EgEm_tight")) {}

edm::ParameterSetDescription l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<std::vector<double>>("wp_pt");
  description.add<std::vector<double>>("wp_PU");
  description.add<std::vector<double>>("wp_Pi");
  description.add<std::vector<double>>("wp_EgEm");
  description.add<std::vector<double>>("wp_EgEm_tight");
  description.add<std::vector<double>>("wp_PFEm");
  return description;
}

l1ct::HgcalClusterDecoderEmulator::MultiClassID::MultiClassID(const edm::ParameterSet &pset) {
  for (auto &wp : pset.getParameter<std::vector<edm::ParameterSet>>("wps"))
    wps_.emplace_back(wp);
  initialize(pset.getParameter<std::string>("model"), pset.getParameter<std::vector<double>>("wp_eta"));
}

edm::ParameterSetDescription l1ct::HgcalClusterDecoderEmulator::MultiClassID::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<std::string>("model");
  description.add<std::vector<double>>("wp_eta");
  description.addVPSet("wps", MultiClassID::WPs::getParameterSetDescription());
  return description;
}

#endif

l1ct::HgcalClusterDecoderEmulator::HgcalClusterDecoderEmulator(
    const std::string &model,
    const std::vector<double> &wp_eta,
    const std::vector<l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs> &wps,
    bool slim,
    const std::string &corrector,
    float correctorEmfMax,
    bool emulateCorrections,
    const std::string &emInterpScenario)
    : slim_{slim},
      multiclass_id_(model, wp_eta, wps),
      corrector_(corrector, correctorEmfMax, false, emulateCorrections, l1tpf::corrector::EmulationMode::Correction),
      emInterpScenario_(setEmInterpScenario(emInterpScenario)) {}

l1ct::HgcalClusterDecoderEmulator::UseEmInterp l1ct::HgcalClusterDecoderEmulator::setEmInterpScenario(
    const std::string &emInterpScenario) {
  if (emInterpScenario == "no")
    return UseEmInterp::No;
  if (emInterpScenario == "emOnly")
    return UseEmInterp::EmOnly;
  if (emInterpScenario == "allKeepHad")
    return UseEmInterp::AllKeepHad;
  if (emInterpScenario == "allKeepTot")
    return UseEmInterp::AllKeepTot;
  throw std::runtime_error("Unknown emInterpScenario: " + emInterpScenario);
}

l1ct::HadCaloObjEmu l1ct::HgcalClusterDecoderEmulator::decode(const l1ct::PFRegionEmu &sector,
                                                              const ap_uint<256> &in,
                                                              bool &valid) const {
  // Word 0
  ap_uint<14> w_pt = in(13, 0);       // 14 bits: 13-0
  ap_uint<14> w_empt = in(27, 14);    // 14 bits: 27-14
  ap_uint<4> w_gctqual = in(31, 28);  //  4 bits: 31-28
  ap_uint<8> w_emf_tot = in(39, 32);  //  8 bits: 39-32
  ap_uint<8> w_emf = in(47, 40);      //  8 bits: 47-40

  // Word 1
  ap_uint<10> w_abseta = in(64 + 9, 64 + 0);   // 10 bits: 9-0
  ap_int<9> w_phi = in(64 + 18, 64 + 10);      //  9 bits: 18-10
  ap_uint<12> w_meanz = in(64 + 30, 64 + 19);  // 12 bits: 30-19

  // Word 2
  ap_uint<6> w_showerlength = in(128 + 18, 128 + 13);      //  6 bits: 18-13
  ap_uint<7> w_sigmazz = in(128 + 38, 128 + 32);           //  7 bits: 38-32
  ap_uint<7> w_sigmaphiphi = in(128 + 45, 128 + 39);       //  7 bits: 45-39
  ap_uint<6> w_coreshowerlength = in(128 + 51, 128 + 46);  //  6 bits: 51-46
  ap_uint<5> w_sigmaetaeta = in(128 + 56, 128 + 52);       //  5 bits: 56-52

  // Word 3
  ap_uint<13> w_sigmarrtot = in(213, 201);  // 13 bits: 213-201 // FIXME: use word3 spare bits

  // Conversion to local (input sector) coordinates
  ap_int<9> w_eta = l1ct::glbeta_t(w_abseta.to_int() * (sector.floatEtaCenter() > 0 ? +1 : -1)) - sector.hwEtaCenter;

  l1ct::HadCaloObjEmu out;
  out.clear();
  if (w_pt == 0)
    return out;
  // if (w_pt == 0 || w_phi > sector.hwPhiHalfWidth || w_phi <= -sector.hwPhiHalfWidth)
  //   return out;
  out.hwPt = w_pt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);
  out.hwEta = w_eta;
  out.hwPhi = w_phi;  // relative to the region center, at calo
  out.hwEmPt = w_empt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);

  if (!slim_) {
    // FIXME: the scaling here is added to the encoded word.
    out.hwSrrTot = w_sigmarrtot * l1ct::srrtot_t(l1ct::Scales::SRRTOT_LSB);
    // We just downscale precision and round to the nearest integer
    out.hwMeanZ = l1ct::meanz_t(std::min(w_meanz.to_int() + 1, (1 << 12) - 1) >> 1);

    // Compute an H/E value: 1/emf - 1 as needed by Composite ID
    // NOTE: this uses the total cluster energy, which is not the case for the eot shower shape!
    // FIXME: could drop once we move the model to the eot fraction
    ap_ufixed<10, 5, AP_RND_CONV, AP_SAT> w_hoe = 256.0 / (w_emf_tot.to_int() + 0.5) - 1;
    out.hwHoe = w_hoe;
  }
  std::vector<MultiClassID::bdt_feature_t> inputs = {w_showerlength,
                                                     w_coreshowerlength,
                                                     w_emf,
                                                     w_abseta - 256,
                                                     w_meanz,  // We use the full resolution here
                                                     w_sigmaetaeta,
                                                     w_sigmaphiphi,
                                                     w_sigmazz};

  // Apply EM interpretation scenario
  if (emInterpScenario_ == UseEmInterp::No) {  // we do not use EM interpretation
    out.hwEmPt = w_emf_tot * out.hwPt / 256;
    // NOTE: only case where hoe consisten with hwEmPt
  } else if (emInterpScenario_ == UseEmInterp::EmOnly) {  // for emID objs, use EM interp as pT and set H = 0
    if (out.hwEmID) {
      out.hwPt = out.hwEmPt;
      out.hwHoe = 0;
    }
  } else if (emInterpScenario_ ==
             UseEmInterp::AllKeepHad) {  // for all objs, replace EM part with EM interp, preserve H
    l1ct::pt_t had_pt = out.hwPt - w_emf_tot * out.hwPt / 256;
    out.hwPt = had_pt + out.hwEmPt;
    // FIXME: we do not recompute hoe for now...
  } else if (emInterpScenario_ ==
             UseEmInterp::AllKeepTot) {  // for all objs, replace EM part with EM interp, preserve pT
    // FIXME: we do not recompute hoe for now...
  }

  bool notPU = multiclass_id_.evaluate(sector, out, inputs);

  // Calibrate pt and set error
  if (corrector_.valid()) {
    float newpt = corrector_.correctedPt(out.floatPt(), out.floatEmPt(), sector.floatGlbEta(out.hwEta));
    out.hwPt = l1ct::Scales::makePtFromFloat(newpt);
    // NOTE: hoe/emfrac are not updated
  }

  // evaluate multiclass model
  valid = notPU && out.hwPt > 0;

  if (!valid) {
    out.clear();
  }

  return out;
}

l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs::WPs(const std::vector<double> &wp_pt,
                                                          const std::vector<double> &wp_PU,
                                                          const std::vector<double> &wp_Pi,
                                                          const std::vector<double> &wp_PFEm,
                                                          const std::vector<double> &wp_EgEm,
                                                          const std::vector<double> &wp_EgEm_tight) {
  assert(wp_PU.size() == wp_Pi.size() && wp_PU.size() == wp_PFEm.size() && wp_PU.size() == wp_EgEm.size() &&
         wp_PU.size() == wp_EgEm_tight.size() && wp_PU.size() == wp_pt.size() + 1);

  for (auto pt : wp_pt)
    this->wp_pt.emplace_back(pt);
  for (auto pu : wp_PU)
    this->wp_PU.emplace_back(pu);
  for (auto pi : wp_Pi)
    this->wp_Pi.emplace_back(pi);
  for (auto egem : wp_EgEm)
    this->wp_EgEm.emplace_back(egem);
  for (auto pfem : wp_PFEm)
    this->wp_PFEm.emplace_back(pfem);
  for (auto egem : wp_EgEm_tight)
    this->wp_EgEm_tight.emplace_back(egem);
}

void l1ct::HgcalClusterDecoderEmulator::MultiClassID::initialize(const std::string &model,
                                                                 const std::vector<double> &wp_eta) {
  assert(wp_eta.size() + 1 == wps_.size());
  for (auto eta : wp_eta)
    wp_eta_.emplace_back(l1ct::Scales::makeGlbEta(eta));

#ifdef CMSSW_GIT_HASH
  auto resolvedFileName = edm::FileInPath(model).fullPath();
#else
  auto resolvedFileName = model;
#endif
  multiclass_bdt_ = std::make_unique<conifer::BDT<bdt_feature_t, bdt_score_t, false>>(resolvedFileName);
}

l1ct::HgcalClusterDecoderEmulator::MultiClassID::MultiClassID(
    const std::string &model,
    const std::vector<double> &wp_eta,
    const std::vector<l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs> &wps)
    : wps_(wps) {
  initialize(model, wp_eta);
}

bool l1ct::HgcalClusterDecoderEmulator::MultiClassID::evaluate(const l1ct::PFRegionEmu &sector,
                                                               l1ct::HadCaloObjEmu &cl,
                                                               const std::vector<bdt_feature_t> &inputs) const {
  auto bdt_score = multiclass_bdt_->decision_function(inputs);  //0 is pu, 1 is pi, 2 is eg
  bdt_score_t raw_scores[3] = {bdt_score[0], bdt_score[1], bdt_score[2]};
  l1ct::id_prob_t sm_scores[3];
  nnet::softmax_stable<bdt_score_t, l1ct::id_prob_t, softmax_config>(raw_scores, sm_scores);

  // softmax_stable<>
  unsigned int eta_bin = 0;
  for (size_t i = wp_eta_.size(); i > 0; --i) {
    if (abs(sector.hwGlbEta(cl.hwEta)) >= wp_eta_[i - 1]) {
      eta_bin = i;
      break;
    }
  }
  const WPs &wps = wps_[eta_bin];
  unsigned int pt_bin = 0;
  for (size_t i = wps.wp_pt.size(); i > 0; --i) {
    if (cl.hwPt >= wps.wp_pt[i - 1]) {
      pt_bin = i;
      break;
    }
  }
  bool passPu = (sm_scores[0] >= wps.wp_PU[pt_bin]);
  // bool passPi = (sm_scores[1] >= wp_Pi_[pt_bin]);  // FIXME: where do we store this?
  bool passPFEm = (sm_scores[2] >= wps.wp_PFEm[pt_bin]);
  bool passEgEm = (sm_scores[2] >= wps.wp_EgEm[pt_bin]);
  bool passEgEm_tight = (sm_scores[2] >= wps.wp_EgEm_tight[pt_bin]);

  // bit 0: PF EM ID
  // bit 1: EG EM ID
  // bit 2: EG Loose ID
  cl.hwEmID = passPFEm | (passEgEm_tight << 1) | (passEgEm << 2);

  cl.hwPiProb = sm_scores[1];
  cl.hwEmProb = sm_scores[2];
  return !passPu;
}

void l1ct::HgcalClusterDecoderEmulator::MultiClassID::softmax(const float rawScores[3], float scores[3]) const {
  // softmax (for now, let's compute the softmax in this code; this needs to be changed to implement on firmware)
  // Softmax implemented in conifer (standalone) is to be integrated here soon; for now, just do "offline" softmax :(
  float denom = exp(rawScores[0]) + exp(rawScores[1]) + exp(rawScores[2]);
  scores[0] = exp(rawScores[0]) / denom;
  scores[1] = exp(rawScores[1]) / denom;
  scores[2] = exp(rawScores[2]) / denom;
}
