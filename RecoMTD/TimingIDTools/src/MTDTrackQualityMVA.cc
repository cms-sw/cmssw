#include "RecoMTD/TimingIDTools/interface/MTDTrackQualityMVA.h"

MTDTrackQualityMVA::MTDTrackQualityMVA(std::string weights_file) {
  std::string options("!Color:Silent");
  std::string method("BDT");

  std::string vars_array[] = {MTDTRACKQUALITYMVA_VARS(VAR_STRING)};
  int nvars = sizeof(vars_array) / sizeof(vars_array[0]);
  vars_.assign(vars_array, vars_array + nvars);

  mva_ = std::make_unique<TMVAEvaluator>();
  mva_->initialize(options, method, weights_file, vars_, spec_vars_, true, false);  //use GBR, GradBoost
}

float MTDTrackQualityMVA::operator()(const reco::TrackRef& trk,
                                     const reco::TrackRef& ext_trk,
                                     const edm::ValueMap<float>& btl_chi2s,
                                     const edm::ValueMap<float>& btl_time_chi2s,
                                     const edm::ValueMap<float>& etl_chi2s,
                                     const edm::ValueMap<float>& etl_time_chi2s,
                                     const edm::ValueMap<float>& tmtds,
                                     const edm::ValueMap<float>& trk_lengths) const {
  const auto& pattern = ext_trk->hitPattern();

  std::map<std::string, float> vars;

  //---training performed only above 0.5 GeV
  constexpr float minPtForMVA = 0.5;
  if (trk->pt() < minPtForMVA)
    return -1;

  //---training performed only for tracks with MTD hits
  if (tmtds[ext_trk] > 0) {
    vars.emplace(vars_[pt], trk->pt());
    vars.emplace(vars_[eta], trk->eta());
    vars.emplace(vars_[phi], trk->phi());
    vars.emplace(vars_[chi2], trk->chi2());
    vars.emplace(vars_[ndof], trk->ndof());
    vars.emplace(vars_[numberOfValidHits], trk->numberOfValidHits());
    vars.emplace(vars_[numberOfValidPixelBarrelHits], pattern.numberOfValidPixelBarrelHits());
    vars.emplace(vars_[numberOfValidPixelEndcapHits], pattern.numberOfValidPixelEndcapHits());
    vars.emplace(vars_[btlMatchChi2], btl_chi2s.contains(ext_trk.id()) ? btl_chi2s[ext_trk] : -1);
    vars.emplace(vars_[btlMatchTimeChi2], btl_time_chi2s.contains(ext_trk.id()) ? btl_time_chi2s[ext_trk] : -1);
    vars.emplace(vars_[etlMatchChi2], etl_chi2s.contains(ext_trk.id()) ? etl_chi2s[ext_trk] : -1);
    vars.emplace(vars_[etlMatchTimeChi2], etl_time_chi2s.contains(ext_trk.id()) ? etl_time_chi2s[ext_trk] : -1);
    vars.emplace(vars_[mtdt], tmtds[ext_trk]);
    vars.emplace(vars_[path_len], trk_lengths[ext_trk]);
    return 1. / (1 + sqrt(2 / (1 + mva_->evaluate(vars, false)) - 1));  //return values between 0-1 (probability)
  } else
    return -1;
}
