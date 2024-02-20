#ifndef DataFormats_HGCalReco_TICLCandidate_h
#define DataFormats_HGCalReco_TICLCandidate_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoHGCal/TICL/interface/commons.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// A TICLCandidate is a lightweight physics object made from one or multiple Tracksters.

class TICLCandidate : public reco::LeafCandidate {
public:
  typedef ticl::Trackster::ParticleType ParticleType;

  TICLCandidate(Charge q, const LorentzVector& p4)
      : LeafCandidate(q, p4), idProbabilities_{}, time_(0.f), timeError_(-1.f), rawEnergy_(0.f) {}

  TICLCandidate() : LeafCandidate(), idProbabilities_{}, time_(0.f), timeError_(-1.f), rawEnergy_(0.f) {}

  TICLCandidate(const edm::Ptr<ticl::Trackster>& trackster)
      : LeafCandidate(),
        tracksters_({trackster}),
        idProbabilities_{},
        time_(trackster->time()),
        timeError_(trackster->timeError()),
        MTDtime_{0.f},
        MTDtimeError_{-1.f},
        rawEnergy_(0.f) {}

  TICLCandidate(const edm::Ptr<reco::Track> trackPtr, const edm::Ptr<ticl::Trackster>& tracksterPtr)
      : LeafCandidate(), tracksters_{}, trackPtr_(trackPtr), time_(0.f), timeError_(-1.f) {
    if (trackPtr_.isNull() and tracksterPtr.isNull())
      edm::LogError("TICLCandidate") << "At least one between track and trackster must be valid\n";

    if (tracksterPtr.isNonnull()) {
      tracksters_.push_back(tracksterPtr);
      auto const& trackster = tracksters_[0].get();
      idProbabilities_ = trackster->id_probabilities();
      if (trackPtr_.isNonnull()) {
        auto pdgId = trackster->isHadronic() ? 211 : 11;
        auto const& tk = trackPtr_.get();
        setPdgId(pdgId * tk->charge());
        setCharge(tk->charge());
        rawEnergy_ = trackster->raw_energy();
        auto const& regrE = trackster->regressed_energy();
        math::XYZTLorentzVector p4(regrE * tk->momentum().unit().x(),
                                   regrE * tk->momentum().unit().y(),
                                   regrE * tk->momentum().unit().z(),
                                   regrE);
        setP4(p4);

      } else {
        auto pdgId = trackster->isHadronic() ? 130 : 22;
        setPdgId(pdgId);
        setCharge(0);
        rawEnergy_ = trackster->raw_energy();
        const float& regrE = trackster->regressed_energy();
        math::XYZTLorentzVector p4(regrE * trackster->barycenter().unit().x(),
                                   regrE * trackster->barycenter().unit().y(),
                                   regrE * trackster->barycenter().unit().z(),
                                   regrE);
        setP4(p4);
      }
    } else {
      //candidate from track only
      auto const& tk = trackPtr_.get();
      setPdgId(211 * tk->charge());
      setCharge(tk->charge());
      const float energy = std::sqrt(tk->p() * tk->p() + ticl::mpion2);
      setRawEnergy(energy);
      math::PtEtaPhiMLorentzVector p4Polar(tk->pt(), tk->eta(), tk->phi(), ticl::mpion);
      setP4(p4Polar);
    }
  }

  inline float time() const { return time_; }
  inline float timeError() const { return timeError_; }

  void setTime(float time, float timeError) {
    time_ = time;
    timeError_ = timeError;
  };

  inline float MTDtime() const { return MTDtime_; }
  inline float MTDtimeError() const { return MTDtimeError_; }

  void setMTDTime(float time, float timeError) {
    MTDtime_ = time;
    MTDtimeError_ = timeError;
  };

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
  // vector of Ptr so Tracksters can come from different collections
  // and there can be derived classes
  std::vector<edm::Ptr<ticl::Trackster> > tracksters_;
  edm::Ptr<reco::Track> trackPtr_;
  // Since it contains multiple tracksters, duplicate the probability interface
  std::array<float, 8> idProbabilities_;

  float time_;
  float timeError_;
  float MTDtime_;
  float MTDtimeError_;
  float rawEnergy_;
};
#endif
