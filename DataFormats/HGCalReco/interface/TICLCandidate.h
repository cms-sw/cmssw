#ifndef DataFormats_HGCalReco_TICLCandidate_h
#define DataFormats_HGCalReco_TICLCandidate_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// A TICLCandidate is a lightweight physics object made from one or multiple Tracksters.

class TICLCandidate : public reco::LeafCandidate {
public:
  typedef ticl::Trackster::ParticleType ParticleType;

  TICLCandidate(Charge q, const LorentzVector& p4)
      : LeafCandidate(q, p4), time_(0.f), timeError_(-1.f), rawEnergy_(0.f), idProbabilities_{} {}

  TICLCandidate() : LeafCandidate(), time_(0.f), timeError_(-1.f), rawEnergy_(0.f), idProbabilities_{} {}

  TICLCandidate(const edm::Ptr<ticl::Trackster>& trackster)
      : LeafCandidate(),
        time_(trackster->time()),
        timeError_(trackster->timeError()),
        rawEnergy_(0.f),
        tracksters_({trackster}),
        idProbabilities_{} {}

  inline float time() const { return time_; }
  inline float timeError() const { return timeError_; }

  void setTime(float time) { time_ = time; };
  void setTimeError(float timeError) { timeError_ = timeError; }

  inline const edm::Ptr<reco::Track> trackPtr() const { return trackPtr_; }
  void setTrackPtr(const edm::Ptr<reco::Track>& trackPtr) { trackPtr_ = trackPtr; }

  inline float rawEnergy() const { return rawEnergy_; }
  void setRawEnergy(float rawEnergy) { rawEnergy_ = rawEnergy; }

  inline const std::vector<edm::Ptr<ticl::Trackster> > tracksters() const { return tracksters_; };

  void setTracksters(const std::vector<edm::Ptr<ticl::Trackster> >& tracksters) { tracksters_ = tracksters; }
  void addTrackster(const edm::Ptr<ticl::Trackster>& trackster) {
    tracksters_.push_back(trackster);
    time_ = trackster->time();
    timeError_ = trackster->timeError();
  }
  // convenience method to return the ID probability for a certain particle type
  inline float id_probability(ParticleType type) const {
    // probabilities are stored in the same order as defined in the ParticleType enum
    return idProbabilities_[(int)type];
  }

  inline const std::array<float, 8>& idProbabilities() const { return idProbabilities_; }

  void zeroProbabilities() {
    for (auto& p : idProbabilities_) {
      p = 0.f;
    }
  }

  void setIdProbabilities(const std::array<float, 8>& idProbs) { idProbabilities_ = idProbs; }
  inline void setIdProbability(ParticleType type, float value) { idProbabilities_[int(type)] = value; }

private:
  float time_;
  float timeError_;
  edm::Ptr<reco::Track> trackPtr_;

  float rawEnergy_;

  // vector of Ptr so Tracksters can come from different collections
  // and there can be derived classes
  std::vector<edm::Ptr<ticl::Trackster> > tracksters_;

  // Since it contains multiple tracksters, duplicate the probability interface
  std::array<float, 8> idProbabilities_;
};
#endif
