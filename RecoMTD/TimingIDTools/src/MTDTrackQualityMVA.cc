#include "RecoMTD/TimingIDTools/interface/MTDTrackQualityMVA.h"

MTDTrackQualityMVA::MTDTrackQualityMVA(std::string weights_file) {
  std::string options("!Color:Silent");
  std::string method("BDT");

  std::string vars_array[] = {MTDTRACKQUALITYMVA_VARS(MTDBDTVAR_STRING)};
  int nvars = sizeof(vars_array) / sizeof(vars_array[0]);
  vars_.assign(vars_array, vars_array + nvars);

  mva_ = std::make_unique<TMVAEvaluator>();
  mva_->initialize(options, method, weights_file, vars_, spec_vars_, true, false);  //use GBR, GradBoost
  //mva_->initialize(options, method, weights_file, vars_, spec_vars_, false, false);  //use TMVA
};

float MTDTrackQualityMVA::operator()(const reco::TrackRef& trk,
                                     const reco::BeamSpot& beamspot,
                                     const edm::ValueMap<int>& npixBarrels,
                                     const edm::ValueMap<int>& npixEndcaps,
                                     const edm::ValueMap<float>& btl_chi2s,
                                     const edm::ValueMap<float>& btl_time_chi2s,
                                     const edm::ValueMap<float>& etl_chi2s,
                                     const edm::ValueMap<float>& etl_time_chi2s,
                                     const edm::ValueMap<float>& tmtds,
                                     const edm::ValueMap<float>& sigmatmtds,
                                     const edm::ValueMap<float>& trk_lengths,
                                     const edm::ValueMap<float>& trk_lhitpos) const {
  std::map<std::string, float> vars;

  static constexpr double etacutREC_ = 3.;   // |eta| < 3
  static constexpr double etacutBTL_ = 1.5;  // |eta| < 1.5 for BTL
  static constexpr double pTcutBTL_ = 0.7;   // PT > 0.7 GeV
  static constexpr double pTcutETL_ = 0.2;   // PT > 0.2 GeV

  //---training performed only for the specified cuts
  if (std::abs(trk->eta()) > etacutREC_)  // max eta cut
    return -1;
  if (std::abs(trk->eta()) <= etacutBTL_ && trk->pt() < pTcutBTL_)  // min PT cut for BTL
    return -1;
  if (std::abs(trk->eta()) > etacutBTL_ && trk->pt() < pTcutETL_)  // min PT cut for ETL
    return -1;

  //---training performed only for tracks with MTD hits
  if (tmtds[trk] > 0) {
    vars.emplace(vars_[int(VarID::Track_pt)], trk->pt());
    vars.emplace(vars_[int(VarID::Track_eta)], trk->eta());
    vars.emplace(vars_[int(VarID::Track_phi)], trk->phi());
    vars.emplace(vars_[int(VarID::Track_dz)], trk->dz(beamspot.position()));
    vars.emplace(vars_[int(VarID::Track_dxy)], trk->dxy(beamspot.position()));
    vars.emplace(vars_[int(VarID::Track_chi2)], trk->chi2());
    vars.emplace(vars_[int(VarID::Track_ndof)], trk->ndof());
    vars.emplace(vars_[int(VarID::Track_npixBarrelValidHits)], npixBarrels[trk]);
    vars.emplace(vars_[int(VarID::Track_npixEndcapValidHits)], npixEndcaps[trk]);
    vars.emplace(vars_[int(VarID::Track_BTLchi2)], btl_chi2s.contains(trk.id()) ? btl_chi2s[trk] : -1);
    vars.emplace(vars_[int(VarID::Track_BTLtime_chi2)], btl_time_chi2s.contains(trk.id()) ? btl_time_chi2s[trk] : -1);
    vars.emplace(vars_[int(VarID::Track_ETLchi2)], etl_chi2s.contains(trk.id()) ? etl_chi2s[trk] : -1);
    vars.emplace(vars_[int(VarID::Track_ETLtime_chi2)], etl_time_chi2s.contains(trk.id()) ? etl_time_chi2s[trk] : -1);
    vars.emplace(vars_[int(VarID::Track_Tmtd)], tmtds[trk]);
    vars.emplace(vars_[int(VarID::Track_sigmaTmtd)], sigmatmtds[trk]);
    vars.emplace(vars_[int(VarID::Track_length)], trk_lengths[trk]);
    vars.emplace(vars_[int(VarID::Track_lHitPos)], trk_lhitpos[trk]);

    return 1. / (1 + sqrt(2 / (1 + mva_->evaluate(vars, false)) - 1));  //return values between 0-1 (probability)
  } else
    return -1;
}
