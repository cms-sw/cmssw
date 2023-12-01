#ifndef RecoTracker_MkFitCore_standalone_RntDumper_RntStructs_h
#define RecoTracker_MkFitCore_standalone_RntDumper_RntStructs_h

// Avoid MkFit includes for now to simpligy pure ROOT builds.
// #include "RecoTracker/MkFitCore/interface/

#include "ROOT/REveVector.hxx"
#include "Math/Point3D.h"
#include "Math/Vector3D.h"

// From CMSSW data formats
/// point in space with cartesian internal representation
typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > XYZPointF;
/// spatial vector with cartesian internal representation
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > XYZVectorF;
/// spatial vector with cylindrical internal representation using pseudorapidity
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > RhoEtaPhiVectorF;
/// spatial vector with polar internal representation
/// WARNING: ROOT dictionary not provided for the type below
// typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> > RThetaPhiVectorF;

using RVec = ROOT::Experimental::REveVector;

struct HeaderLayer {
  int event, iter_idx, iter_algo, eta_region, layer;
  float qb_min, qb_max;  // qbar layer limits, r for barrel, z for endcap
  bool is_barrel, is_pix, is_stereo;

  HeaderLayer() = default;
  HeaderLayer& operator=(const HeaderLayer&) = default;
};

struct State {
  RVec pos, mom;

  State() = default;
  State& operator=(const State&) = default;
};

struct PropState : public State {
  float dalpha;  // helix angle during propagation
  int fail_flag;

  PropState() = default;
  PropState& operator=(const PropState&) = default;
};

struct SimSeedInfo {
  State s_sim;
  State s_seed;
  int sim_lbl, seed_lbl, seed_idx;
  int n_hits, n_match;
  bool has_sim = false;

  float good_frac() const { return (float)n_match / n_hits; }

  SimSeedInfo() = default;
  SimSeedInfo& operator=(const SimSeedInfo&) = default;
};

struct BinSearch {
  float phi, dphi, q, dq;
  short unsigned int p1, p2, q1, q2;
  short int wsr;
  bool wsr_in_gap;
  bool has_nans = false;

  bool nan_check();

  BinSearch() = default;
  BinSearch& operator=(const BinSearch&) = default;
};

struct CandInfo {
  SimSeedInfo ssi;
  State s_ctr;
  PropState ps_min, ps_max;
  BinSearch bso;
  BinSearch bsn;
  bool has_nans = false;

  CandInfo(const SimSeedInfo& s, const State& c) : ssi(s), s_ctr(c) {}

  void nan_check();

  CandInfo() = default;
  CandInfo& operator=(const CandInfo&) = default;
};

struct FailedPropInfo {
  SimSeedInfo ssi;
  State s_prev;
  State s_final;
  bool has_nans = false;

  FailedPropInfo(const SimSeedInfo& s, const State& p, const State& f) : ssi(s), s_prev(p), s_final(f) {}

  void nan_check();

  FailedPropInfo() = default;
  FailedPropInfo& operator=(const FailedPropInfo&) = default;
};

#endif
