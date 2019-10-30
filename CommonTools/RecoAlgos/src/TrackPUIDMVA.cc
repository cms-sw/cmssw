
#include "CommonTools/RecoAlgos/interface/TrackPUIDMVA.h"

TrackPUIDMVA::TrackPUIDMVA(std::string weights_file) {
  vars_.push_back("pt");
  vars_.push_back("eta");
  vars_.push_back("phi");
  vars_.push_back("chi2");
  vars_.push_back("ndof");
  vars_.push_back("numberOfValidHits");
  vars_.push_back("numberOfValidPixelBarrelHits");
  vars_.push_back("numberOfValidPixelEndcapHits");
  vars_.push_back("btlMatchChi2");
  vars_.push_back("btlMatchTimeChi2");
  vars_.push_back("etlMatchChi2");
  vars_.push_back("etlMatchTimeChi2");
  vars_.push_back("mtdt");
  vars_.push_back("path_len");

  std::string options("!Color:Silent");
  std::string method("BDT");

  mva_ = std::make_unique<TMVAEvaluator>();
  mva_->initialize(options, method, weights_file, vars_, spec_vars_, true, false);
}

float TrackPUIDMVA::operator()(const reco::TrackRef& trk,
                               const reco::TrackRef& ext_trk,
                               edm::ValueMap<float>& btl_chi2s,
                               edm::ValueMap<float>& btl_time_chi2s,
                               edm::ValueMap<float>& etl_chi2s,
                               edm::ValueMap<float>& etl_time_chi2s,
                               edm::ValueMap<float>& tmtds,
                               edm::ValueMap<float>& trk_lengths) const {
  const auto& pattern = ext_trk->hitPattern();

  std::map<std::string, float> vars;

  vars.emplace(vars_[0], trk->pt());
  vars.emplace(vars_[1], trk->eta());
  vars.emplace(vars_[2], trk->phi());
  vars.emplace(vars_[3], trk->chi2());
  vars.emplace(vars_[4], trk->ndof());
  vars.emplace(vars_[5], trk->numberOfValidHits());
  vars.emplace(vars_[6], pattern.numberOfValidPixelBarrelHits());
  vars.emplace(vars_[7], pattern.numberOfValidPixelEndcapHits());
  vars.emplace(vars_[8], btl_chi2s.contains(ext_trk.id()) ? btl_chi2s[ext_trk] : -1);
  vars.emplace(vars_[9], btl_time_chi2s.contains(ext_trk.id()) ? btl_time_chi2s[ext_trk] : -1);
  vars.emplace(vars_[10], etl_chi2s.contains(ext_trk.id()) ? etl_chi2s[ext_trk] : -1);
  vars.emplace(vars_[11], etl_time_chi2s.contains(ext_trk.id()) ? etl_time_chi2s[ext_trk] : -1);
  vars.emplace(vars_[12], tmtds[ext_trk]);
  vars.emplace(vars_[13], trk_lengths[ext_trk]);

  return mva_->evaluate(vars, false);
}
