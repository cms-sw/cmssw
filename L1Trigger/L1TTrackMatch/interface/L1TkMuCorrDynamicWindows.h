#ifndef L1Trigger_L1TTrackMatch_L1TKMUCORRDYNAMICWINDOWS_H
#define L1Trigger_L1TTrackMatch_L1TKMUCORRDYNAMICWINDOWS_H

#include "TFile.h"
#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/Math/interface/angle_units.h"

#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

class L1TkMuCorrDynamicWindows {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  L1TkMuCorrDynamicWindows(const std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi);
  L1TkMuCorrDynamicWindows(
      const std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi, TFile* fIn_theta_S1, TFile* fIn_phi_S1);
  ~L1TkMuCorrDynamicWindows() {}
  std::vector<int> find_match(
      const EMTFTrackCollection& l1mus,
      const L1TTTrackCollectionType& l1trks);  // gives a vector with the idxs of muons for each L1TTT
  std::vector<int> find_match_stub(
      const EMTFHitCollection& l1mus,
      const L1TTTrackCollectionType& l1trks,
      const int& station,
      bool requireBX0 = true);  // gives a vector with the idxs of muon stubs from station "station" for each L1TTT

  // ------------------------------
  static std::vector<double> prepare_corr_bounds(const string& fname, const string& hname);

  void set_safety_factor(float sf_l, float sf_h) {
    safety_factor_l_ = sf_l;
    safety_factor_h_ = sf_h;
  }
  void set_sf_initialrelax(float sf_l, float sf_h) {
    initial_sf_l_ = sf_l;
    initial_sf_h_ = sf_h;
  }
  void set_relaxation_pattern(float pt_start, float pt_end) {
    pt_start_ = pt_start;
    pt_end_ = pt_end;
  }
  void set_safety_factor(float sf) { set_safety_factor(sf, sf); }
  void set_sf_initialrelax(float sf) { set_sf_initialrelax(sf, sf); }
  void set_do_relax_factor(bool val) { do_relax_factor_ = val; }

  void set_do_trk_qual_presel(bool val) { track_qual_presel_ = val; }

  // setters for trk
  void set_n_trk_par(int val) { nTrkPars_ = val; }
  void set_min_trk_p(float val) { min_trk_p_ = val; }
  void set_max_trk_aeta(float val) { max_trk_aeta_ = val; }
  void set_max_trk_chi2(float val) { max_trk_chi2_ = val; }
  void set_min_trk_nstubs(int val) { min_trk_nstubs_ = val; }

  // getters for trk
  const int n_trk_par() { return nTrkPars_; }
  const float min_trk_p() { return min_trk_p_; }
  const float max_trk_aeta() { return max_trk_aeta_; }
  const float max_trk_chi2() { return max_trk_chi2_; }
  const int min_trk_nstubs() { return min_trk_nstubs_; }

private:
  int findBin(double val);

  // resolves ambiguities to give max 1 tkmu per EMTF
  // if a pointer to narbitrated is passed, this vector is filled with the number of tracks arbitrated that were matched to the same EMTF
  std::vector<int> make_unique_coll(const unsigned int& l1musSize,
                                    const L1TTTrackCollectionType& l1trks,
                                    const std::vector<int>& matches);

  // converters
  double eta_to_theta(double x) {
    //  give theta in rad
    return (2. * atan(exp(-1. * x)));
  }

  double to_mpio2_pio2(double x) {
    //  put the angle in radians between -pi/2 and pi/2
    while (x >= 0.5 * M_PI)
      x -= M_PI;
    while (x < -0.5 * M_PI)
      x += M_PI;
    return x;
  }

  double sf_progressive(double x, double xstart, double xstop, double ystart, double ystop) {
    if (x < xstart)
      return ystart;
    if (x >= xstart && x < xstop)
      return ystart + (x - xstart) * (ystop - ystart) / (xstop - xstart);
    return ystop;
  }

  int nbins_;                   // counts the number of MatchWindow = bounds_.size() - 1
  std::vector<double> bounds_;  // counts the boundaries of the MatchWindow (in eta/theta)
  std::vector<MuMatchWindow> wdws_theta_;
  std::vector<MuMatchWindow> wdws_phi_;
  std::vector<MuMatchWindow> wdws_theta_S1_;
  std::vector<MuMatchWindow> wdws_phi_S1_;
  float safety_factor_l_;   // increase the lower theta/phi threshold by this fractions
  float safety_factor_h_;   // increase the upper theta/phi threshold by this fractions
  float initial_sf_l_;      // the start of the relaxation
  float initial_sf_h_;      // the start of the relaxation
  float pt_start_;          // the relaxation of the threshold
  float pt_end_;            // the relaxation of the threshold
  bool do_relax_factor_;    // true if applying the linear relaxation
  bool track_qual_presel_;  // if true, apply the track preselection

  // trk configurable params
  int nTrkPars_;        // 4
  float min_trk_p_;     // 3.5
  float max_trk_aeta_;  // 2.5
  float max_trk_chi2_;  // 100
  int min_trk_nstubs_;  // 4
};

#endif
