#ifndef DataFormats_HGCalReco_TICLCandidate_h
#define DataFormats_HGCalReco_TICLCandidate_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// A TICLCandidate is a trackster with reconstructed energy and position.

namespace ticl {
class TICLCandidate : public reco::LeafCandidate {
 public:
  TICLCandidate(Charge q, const LorentzVector& p4) : LeafCandidate(q, p4), time_(0.f), time_error_(-1.f) {
  }

  TICLCandidate() : LeafCandidate(), time_(0.f), time_error_(-1.f) {
  }

  inline float time() const { return time_; }
  inline float time_error() const { return time_error_; }

  void set_time(float time) { time_ = time; };
  void set_time_error(float time_error) { time_error_ = time_error; }

  inline const reco::TrackRef track_ref() const { return track_ref_; }
  void set_track_ref(const reco::TrackRef& track_ref) {
    track_ref_ = track_ref;
  }

  inline const std::vector<edm::Ptr<Trackster> > tracksters() const {
    return tracksters_;
  };

  void set_tracksters(const std::vector<edm::Ptr<Trackster> >& tracksters) {
    tracksters_ = tracksters;
  }
  void add_trackster(const edm::Ptr<Trackster>& trackster) {
    tracksters_.push_back(trackster);
  }

  // convenience methods to return certain id probabilities
  inline float photon_probability() const { return id_probabilities_[0]; }
  inline float electron_probability() const { return id_probabilities_[1]; }
  inline float muon_probability() const { return id_probabilities_[2]; }
  inline float charged_hadron_probability() const { return id_probabilities_[3]; }
  inline float neutral_hadron_probability() const { return id_probabilities_[4]; }
  inline float ambiguous_probability() const { return id_probabilities_[5]; }
  inline float unknown_probability() const { return id_probabilities_[6]; }

  void set_id_probabilities(const std::array<float, 7>& id_probs) {
    id_probabilities_ = id_probs;
  }

 private:
  float time_;
  float time_error_;
  reco::TrackRef track_ref_;  // edm::Ref so it's possible to create an edm::Ref
                              // for the PFCandidate

  // SOME_KIND_OF_SEED for Felice

  // vector of Ptr so Tracksters can come from different collections
  // and there can be derived classes
  std::vector<edm::Ptr<Trackster> > tracksters_;

  // Since it contains multiple tracksters, duplicate the probability interface
  std::array<float, 7> id_probabilities_;
};
}  // namespace ticl
#endif
