#ifndef L1Trigger_L1TTrackMatch_L1TKMUMANTRA_H
#define L1Trigger_L1TTrackMatch_L1TKMUMANTRA_H

/*
** class  : GenericDataFormat
** author : L.Cadamuro (UF)
** date   : 4/11/2019
** brief  : very generic structs to be used as inputs to the correlator
**        : to make sure that Mantra can handle muons and tracks from all the detectors
*/

namespace L1TkMuMantraDF {

  struct track_df {
    double pt;     // GeV
    double eta;    // rad, -inf / +inf
    double theta;  // rad, 0 -> +90-90
    double phi;    // rad, -pi / + pi
    int nstubs;    //
    double chi2;   //
    int charge;    // -1. +1
  };

  struct muon_df {
    double pt;     // GeV
    double eta;    // rad, -inf / +inf
    double theta;  // rad, 0 -> +90-90
    double phi;    // rad, -pi / + pi
    int charge;    // -1. +1
  };
}  // namespace L1TkMuMantraDF

/*
** class  : L1TkMuMantra
** author : L.Cadamuro (UF)
** date   : 4/11/2019
** brief  : correlates muons and tracks using pre-encoded windows
*/

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"
#include <cmath>

#include "TFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/angle_units.h"

class L1TkMuMantra {
public:
  L1TkMuMantra(const std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi, std::string name);
  ~L1TkMuMantra(){};

  // returns a vector with the same size of muons, each with an index to the matched L1 track, or -1 if no match is found
  std::vector<int> find_match(const std::vector<L1TkMuMantraDF::track_df>& tracks,
                              const std::vector<L1TkMuMantraDF::muon_df>& muons);

  void test(double eta, double pt);

  void relax_windows(double& low, double cent, double& high);  // will modify low and high

  void set_safety_factor(float sf_l, float sf_h) {
    safety_factor_l_ = sf_l;
    safety_factor_h_ = sf_h;
    if (verbosity_ > 0)
      LogTrace("L1TkMuMantra") << name_ << " safety factor LOW is " << safety_factor_l_ << std::endl;
    if (verbosity_ > 0)
      LogTrace("L1TkMuMantra") << name_ << " safety factor HIGH is " << safety_factor_h_ << std::endl;
  }

  int sign(double x) {
    if (x == 0)
      return 1;
    return (0 < x) - (x < 0);
  }

  void setArbitrationType(std::string type);  // MaxPt, MinDeltaPt

  // static functions, meant to be used from outside to interface with MAnTra
  static std::vector<double> prepare_corr_bounds(std::string fname, std::string hname);

  // converters
  static double eta_to_theta(double x) {
    //  give theta in rad
    return (2. * atan(exp(-1. * x)));
  }

  static double to_mpio2_pio2(double x) {
    //  put the angle in radians between -pi/2 and pi/2
    while (x >= 0.5 * M_PI)
      x -= M_PI;
    while (x < -0.5 * M_PI)
      x += M_PI;
    return x;
  }

private:
  int findBin(double val);

  std::string name_;

  int nbins_;                   // counts the number of MuMatchWindow = bounds_.size() - 1
  std::vector<double> bounds_;  // counts the boundaries of the MuMatchWindow (in eta/theta)
  std::vector<MuMatchWindow> wdws_theta_;
  std::vector<MuMatchWindow> wdws_phi_;

  int min_nstubs = 4;     // >= min_nstubs
  double max_chi2 = 100;  // < max_chi2

  float safety_factor_l_;  // increase the lower theta/phi threshold by this fractions w.r.t. the center
  float safety_factor_h_;  // increase the upper theta/phi threshold by this fractions w.r.t. the center

  enum sortParType {
    kMaxPt,      // pick the highest pt track matched
    kMinDeltaPt  // pick the track with the smallest pt difference w.r.t the muon
  };

  sortParType sort_type_;

  int verbosity_ = 0;
};

#endif  // L1TKMUMANTRA_H
