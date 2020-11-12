#include "L1Trigger/L1TTrackMatch/interface/L1TkMuCorrDynamicWindows.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/angle_units.h"

// ROOT includes
#include "TH1.h"
#include "TH2.h"

L1TkMuCorrDynamicWindows::L1TkMuCorrDynamicWindows(const std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi)
    : wdws_theta_(bounds.size() - 1, MuMatchWindow()),
      wdws_phi_(bounds.size() - 1, MuMatchWindow()),
      wdws_theta_S1_(bounds.size() - 1, MuMatchWindow()),
      wdws_phi_S1_(bounds.size() - 1, MuMatchWindow()) {
  set_safety_factor(0.5);
  set_sf_initialrelax(0.0);
  set_relaxation_pattern(2.0, 6.0);
  set_do_relax_factor(true);

  track_qual_presel_ = true;

  nbins_ = bounds.size() - 1;
  bounds_ = bounds;

  // now load in memory the TF1 fits

  for (int ib = 0; ib < nbins_; ++ib) {
    std::string wdn;
    std::string nml;
    std::string nmh;
    TF1* fl;
    TF1* fh;

    // Station 2
    wdn = std::string("wdw_theta_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_theta->Get(nml.c_str());
    fh = (TF1*)fIn_theta->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_theta_.at(ib).SetName(wdn);
    wdws_theta_.at(ib).SetLower(fl);
    wdws_theta_.at(ib).SetUpper(fh);

    wdn = std::string("wdw_phi_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_phi->Get(nml.c_str());
    fh = (TF1*)fIn_phi->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_phi_.at(ib).SetName(wdn);
    wdws_phi_.at(ib).SetLower(fl);
    wdws_phi_.at(ib).SetUpper(fh);
  }
}

L1TkMuCorrDynamicWindows::L1TkMuCorrDynamicWindows(
    const std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi, TFile* fIn_theta_S1, TFile* fIn_phi_S1)
    : wdws_theta_(bounds.size() - 1, MuMatchWindow()),
      wdws_phi_(bounds.size() - 1, MuMatchWindow()),
      wdws_theta_S1_(bounds.size() - 1, MuMatchWindow()),
      wdws_phi_S1_(bounds.size() - 1, MuMatchWindow()) {
  set_safety_factor(0.0);
  set_sf_initialrelax(0.0);
  set_relaxation_pattern(2.0, 6.0);
  set_do_relax_factor(true);

  track_qual_presel_ = true;

  nbins_ = bounds.size() - 1;
  bounds_ = bounds;

  // now load in memory the TF1 fits

  for (int ib = 0; ib < nbins_; ++ib) {
    std::string wdn;
    std::string nml;
    std::string nmh;
    TF1* fl;
    TF1* fh;

    // Station 2
    wdn = std::string("wdw_theta_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_theta->Get(nml.c_str());
    fh = (TF1*)fIn_theta->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_theta_.at(ib).SetName(wdn);
    wdws_theta_.at(ib).SetLower(fl);
    wdws_theta_.at(ib).SetUpper(fh);

    wdn = std::string("wdw_phi_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_phi->Get(nml.c_str());
    fh = (TF1*)fIn_phi->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_phi_.at(ib).SetName(wdn);
    wdws_phi_.at(ib).SetLower(fl);
    wdws_phi_.at(ib).SetUpper(fh);

    // Station 1 - MW's don't have to exist for TkMuon correlator
    // It is only needed for TkMuStub
    wdn = std::string("wdw_theta_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_theta_S1->Get(nml.c_str());
    fh = (TF1*)fIn_theta_S1->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_theta_S1_.at(ib).SetName(wdn);
    wdws_theta_S1_.at(ib).SetLower(fl);
    wdws_theta_S1_.at(ib).SetUpper(fh);

    wdn = std::string("wdw_phi_") + std::to_string(ib + 1);
    nml = std::string("fit_low_") + std::to_string(ib + 1);
    nmh = std::string("fit_high_") + std::to_string(ib + 1);
    fl = (TF1*)fIn_phi_S1->Get(nml.c_str());
    fh = (TF1*)fIn_phi_S1->Get(nmh.c_str());
    if (fl == nullptr || fh == nullptr)
      throw cms::Exception("L1TkMuCorrDynamicWindows")
          << "TF1 named " << nml << " or " << nmh << " not found in file " << fIn_theta->GetName() << ".\n";
    wdws_phi_S1_.at(ib).SetName(wdn);
    wdws_phi_S1_.at(ib).SetLower(fl);
    wdws_phi_S1_.at(ib).SetUpper(fh);
  }
}

int L1TkMuCorrDynamicWindows::findBin(double val) {
  // not the most efficient, nor the most elegant implementation for now
  if (val < bounds_.at(0))
    return 0;
  if (val >= bounds_.back())
    return (nbins_ - 1);  // i.e. bounds_size() -2

  for (uint ib = 0; ib < bounds_.size() - 1; ++ib) {
    if (val >= bounds_.at(ib) && val < bounds_.at(ib + 1))
      return ib;
  }

  //"Something strange happened at val.
  throw cms::Exception("L1TkMuCorrDynamicWindows") << "Can't find bin.\n";
  return 0;
}

std::vector<int> L1TkMuCorrDynamicWindows::find_match(const EMTFTrackCollection& l1mus,
                                                      const L1TTTrackCollectionType& l1trks) {
  std::vector<int> out(l1trks.size());
  for (auto l1trkit = l1trks.begin(); l1trkit != l1trks.end(); ++l1trkit) {
    float trk_pt = l1trkit->momentum().perp();
    float trk_p = l1trkit->momentum().mag();
    float trk_aeta = std::abs(l1trkit->momentum().eta());
    float trk_theta = to_mpio2_pio2(eta_to_theta(l1trkit->momentum().eta()));
    float trk_phi = l1trkit->momentum().phi();
    int trk_charge = (l1trkit->rInv() > 0 ? 1 : -1);

    // porting some selections from the MuonTrackCorr finder
    // https://github.com/cms-l1t-offline/cmssw/blob/l1t-phase2-932-v1.6/L1Trigger/L1TTrackMatch/plugins/L1TkMuonProducer.cc#L264
    // in future, make preselections confiuguable
    bool reject_trk = false;
    if (trk_p < min_trk_p_)
      reject_trk = true;
    if (trk_aeta > max_trk_aeta_)
      reject_trk = true;
    if (track_qual_presel_) {
      float l1tk_chi2 = l1trkit->chi2();
      int l1tk_nstubs = l1trkit->getStubRefs().size();
      if (l1tk_chi2 >= max_trk_chi2_)
        reject_trk = true;
      if (l1tk_nstubs < min_trk_nstubs_)
        reject_trk = true;
    }

    int ibin = findBin(trk_aeta);

    std::vector<std::tuple<float, float, int>> matched;  // dtheta, dphi, idx
    // loop on muons to see which match
    for (auto l1muit = l1mus.begin(); l1muit != l1mus.end(); ++l1muit) {
      // match only muons in the central bx - as the track collection refers anyway to bx 0 only
      if (l1muit->BX() != 0)
        continue;

      // putting everything in rad
      float emtf_theta = to_mpio2_pio2(eta_to_theta(l1muit->Eta()));
      float emtf_phi = angle_units::operators::convertDegToRad(l1muit->Phi_glob());

      float dtheta = std::abs(emtf_theta - trk_theta);
      float dphi = reco::deltaPhi(emtf_phi, trk_phi);
      float adphi = std::abs(dphi);

      double sf_l;
      double sf_h;
      if (do_relax_factor_) {
        sf_l = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_l_, safety_factor_l_);
        sf_h = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_h_, safety_factor_h_);
      } else {
        sf_l = safety_factor_l_;
        sf_h = safety_factor_h_;
      }

      if (
          // emtf_theta * trk_theta > 0 &&
          dtheta > (1 - sf_l) * wdws_theta_.at(ibin).bound_low(trk_pt) &&
          dtheta <= (1 + sf_h) * wdws_theta_.at(ibin).bound_high(trk_pt) &&
          adphi > (1 - sf_l) * wdws_phi_.at(ibin).bound_low(trk_pt) &&
          adphi <= (1 + sf_h) * wdws_phi_.at(ibin).bound_high(trk_pt) && dphi * trk_charge < 0 &&  // sign requirement
          // rndm > 0.5
          true)
        matched.emplace_back(dtheta, adphi, std::distance(l1mus.begin(), l1muit));
    }

    if (reject_trk)
      matched.clear();  // quick fix - to be optimised to avoid the operations above

    if (matched.empty())
      out.at(std::distance(l1trks.begin(), l1trkit)) = -1;
    else {
      std::sort(matched.begin(), matched.end());  // closest in theta, then in phi
      out.at(std::distance(l1trks.begin(), l1trkit)) = std::get<2>(matched.at(0));
    }
  }

  // now convert out to a unique set
  return make_unique_coll(l1mus.size(), l1trks, out);
}

std::vector<int> L1TkMuCorrDynamicWindows::find_match_stub(const EMTFHitCollection& l1mus,
                                                           const L1TTTrackCollectionType& l1trks,
                                                           const int& station,
                                                           bool requireBX0) {
  std::vector<int> out(l1trks.size());
  for (auto l1trkit = l1trks.begin(); l1trkit != l1trks.end(); ++l1trkit) {
    float trk_pt = l1trkit->momentum().perp();
    float trk_p = l1trkit->momentum().mag();
    float trk_aeta = std::abs(l1trkit->momentum().eta());
    float trk_theta = to_mpio2_pio2(eta_to_theta(l1trkit->momentum().eta()));
    float trk_phi = l1trkit->momentum().phi();
    int trk_charge = (l1trkit->rInv() > 0 ? 1 : -1);

    // porting some selections from the MuonTrackCorr finder
    // https://github.com/cms-l1t-offline/cmssw/blob/l1t-phase2-932-v1.6/L1Trigger/L1TTrackMatch/plugins/L1TkMuonProducer.cc#L264
    // for future: make preselections confiuguable
    bool reject_trk = false;
    if (trk_p < min_trk_p_)
      reject_trk = true;
    if (trk_aeta > max_trk_aeta_)
      reject_trk = true;
    if (track_qual_presel_) {
      float l1tk_chi2 = l1trkit->chi2();
      int l1tk_nstubs = l1trkit->getStubRefs().size();
      if (l1tk_chi2 >= max_trk_chi2_)
        reject_trk = true;
      if (l1tk_nstubs < min_trk_nstubs_)
        reject_trk = true;
    }

    int ibin = findBin(trk_aeta);

    std::vector<std::tuple<float, float, int>> matched;  // dtheta, dphi, idx
    // loop on stubs to see which match
    for (auto l1muit = l1mus.begin(); l1muit != l1mus.end(); ++l1muit) {
      if (!(l1muit->Is_CSC() || l1muit->Is_RPC()))
        continue;

      int hit_station = l1muit->Station();
      // match only stubs in the central bx - as the track collection refers anyway to bx 0 only
      if (requireBX0 && l1muit->BX() != 0)
        continue;

      // allow only track matching to stubs from the given station, station= 1,2,3,4
      if (station < 5 && hit_station != station)
        continue;
      // in case of station=12 allow track matching to stubs from either station 1 or 2.
      else if (station == 12 && hit_station > 2)  // good for tkMuStub12
        continue;
      // in case of station=123 allow track matching to stubs from either station 1, 2, or 3.
      else if (station == 123 && hit_station > 3)  // good for tkMuStub123
        continue;
      // in case of station=1234 allow track matching to stubs from either station 1, 2, 3, or 4.
      else if (station == 1234 && hit_station > 4)  // good for tkMuStub1234
        continue;

      float emtf_theta = to_mpio2_pio2(eta_to_theta(l1muit->Eta_sim()));
      float emtf_phi = angle_units::operators::convertDegToRad(l1muit->Phi_sim());

      float dtheta = std::abs(emtf_theta - trk_theta);
      float dphi = reco::deltaPhi(emtf_phi, trk_phi);
      float adphi = std::abs(dphi);

      double sf_l;
      double sf_h;
      if (do_relax_factor_) {
        sf_l = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_l_, safety_factor_l_);
        sf_h = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_h_, safety_factor_h_);
      } else {
        sf_l = safety_factor_l_;
        sf_h = safety_factor_h_;
      }

      if (hit_station == 1 &&
          //if hit in station 1 use these  matching windows for checking

          dtheta > (1 - sf_l) * wdws_theta_S1_.at(ibin).bound_low(trk_pt) &&
          dtheta <= (1 + sf_h) * wdws_theta_S1_.at(ibin).bound_high(trk_pt) &&
          adphi > (1 - sf_l) * wdws_phi_S1_.at(ibin).bound_low(trk_pt) &&
          adphi <= (1 + sf_h) * wdws_phi_S1_.at(ibin).bound_high(trk_pt) &&
          dphi * trk_charge < 0 &&  // sign requirement
          true)
        matched.emplace_back(dtheta, adphi, std::distance(l1mus.begin(), l1muit));

      if (hit_station == 2 && dtheta > (1 - sf_l) * wdws_theta_.at(ibin).bound_low(trk_pt) &&
          dtheta <= (1 + sf_h) * wdws_theta_.at(ibin).bound_high(trk_pt) &&
          adphi > (1 - sf_l) * wdws_phi_.at(ibin).bound_low(trk_pt) &&
          adphi <= (1 + sf_h) * wdws_phi_.at(ibin).bound_high(trk_pt) && dphi * trk_charge < 0 &&  // sign requirement
          // rndm > 0.5
          true)
        matched.emplace_back(dtheta, adphi, std::distance(l1mus.begin(), l1muit));
    }

    if (reject_trk)
      matched.clear();  // quick fix - to be optimised to avoid the operations above

    if (matched.empty())
      out.at(std::distance(l1trks.begin(), l1trkit)) = -1;
    else {
      std::sort(matched.begin(), matched.end());  // closest in theta, then in phi
      out.at(std::distance(l1trks.begin(), l1trkit)) = std::get<2>(matched.at(0));
    }
  }

  // return out;

  // now convert out to a unique set
  auto unique_out = make_unique_coll(l1mus.size(), l1trks, out);

  return unique_out;
}

std::vector<int> L1TkMuCorrDynamicWindows::make_unique_coll(const unsigned int& l1musSize,
                                                            const L1TTTrackCollectionType& l1trks,
                                                            const std::vector<int>& matches) {
  std::vector<int> out(matches.size(), -1);

  std::vector<std::vector<int>> macthed_to_emtf(l1musSize,
                                                std::vector<int>(0));  // one vector of matched trk idx per EMTF

  for (unsigned int itrack = 0; itrack < matches.size(); ++itrack) {
    int iemtf = matches.at(itrack);
    if (iemtf < 0)
      continue;
    macthed_to_emtf.at(iemtf).push_back(itrack);
  }

  std::function<bool(int, int, const L1TTTrackCollectionType&, int)> track_less_than_proto =
      [](int idx1, int idx2, const L1TTTrackCollectionType& l1trkcoll, int nTrackParams) {
        float pt1 = l1trkcoll.at(idx1).momentum().perp();
        float pt2 = l1trkcoll.at(idx2).momentum().perp();
        return (pt1 < pt2);
      };

  // // and binds to accept only 2 params
  std::function<bool(int, int)> track_less_than =
      std::bind(track_less_than_proto, std::placeholders::_1, std::placeholders::_2, l1trks, nTrkPars_);

  for (unsigned int iemtf = 0; iemtf < macthed_to_emtf.size(); ++iemtf) {
    std::vector<int>& thisv = macthed_to_emtf.at(iemtf);
    if (thisv.empty())
      continue;

    std::sort(thisv.begin(), thisv.end(), track_less_than);

    // copy to the output
    int best_trk = thisv.back();
    out.at(best_trk) = iemtf;
  }

  return out;
}

std::vector<double> L1TkMuCorrDynamicWindows::prepare_corr_bounds(const string& fname, const string& hname) {
  // find the boundaries of the match windoww
  TFile* fIn = TFile::Open(fname.c_str());
  if (fIn == nullptr) {
    throw cms::Exception("L1TkMuMantra") << "Can't find file " << fname << " to derive bounds.\n";
  }
  TH2* h_test = (TH2*)fIn->Get(hname.c_str());
  if (h_test == nullptr) {
    throw cms::Exception("L1TkMuCorrDynamicWindows")
        << "Can't find histo " << hname << " in file " << fname << " to derive bounds.\n";
  }

  int nbds = h_test->GetNbinsY() + 1;
  vector<double> bounds(nbds);
  for (int ib = 0; ib < nbds; ++ib) {
    bounds.at(ib) = h_test->GetYaxis()->GetBinLowEdge(ib + 1);
  }
  fIn->Close();
  return bounds;
}
