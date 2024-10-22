#ifndef RecoEgamma_EgammaTools_LowPtConversion_h
#define RecoEgamma_EgammaTools_LowPtConversion_h

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"

class LowPtConversion {
public:
  LowPtConversion() = default;
  ~LowPtConversion() = default;

  bool wpOpen() const;   // Matched to any conversion (without selections)
  bool wpLoose() const;  // Nancy's baseline selections for conversions
  bool wpTight() const;  // Nancy's selection for analysis of conversions

  void addUserVars(pat::Electron& ele) const;       // adds minimal set of flags to electron userData
  void addExtraUserVars(pat::Electron& ele) const;  // adds all variables to electron userData

  bool match(const reco::BeamSpot& beamSpot, const reco::ConversionCollection& conversions, const pat::Electron& ele);

  static float mee(float ipx1, float ipy1, float ipz1, float ipx2, float ipy2, float ipz2);

private:
  // quality
  bool valid_ = false;
  float chi2prob_ = -1.;
  bool quality_high_purity_ = false;
  bool quality_high_efficiency_ = false;

  // tracks
  uint ntracks_ = 0;
  float min_trk_pt_ = -1.;
  int ilead_ = -1;
  int itrail_ = -1;

  // displacement
  float l_xy_ = -1.;
  float vtx_radius_ = -1.;

  // invariant mass
  float mass_from_conv_ = -1.;
  float mass_from_Pin_ = -1.;
  float mass_before_fit_ = -1.;
  float mass_after_fit_ = -1.;

  // hits before vertex
  uint lead_nhits_before_vtx_ = 0;
  uint trail_nhits_before_vtx_ = 0;
  uint max_nhits_before_vtx_ = 0;
  uint sum_nhits_before_vtx_ = 0;
  int delta_expected_nhits_inner_ = 0;

  // opening angle
  float delta_cot_from_Pin_ = -1.;

  // match?
  bool matched_ = false;
  edm::RefToBase<reco::Track> matched_lead_;
  edm::RefToBase<reco::Track> matched_trail_;
};

#endif  // RecoEgamma_EgammaTools_LowPtConversion_h
