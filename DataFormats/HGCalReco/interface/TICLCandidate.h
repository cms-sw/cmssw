#ifndef DataFormats_HGCalReco_TICLCandidate_h
#define DataFormats_HGCalReco_TICLCandidate_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Common.h"

#include <array>
#include <cmath>
#include <cstdint>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// A TICLCandidate is a lightweight physics object made from one or multiple Tracksters.

namespace io_v1 {
  class TICLCandidate : public reco::LeafCandidate {
  public:
    typedef ticl::Trackster::ParticleType ParticleType;

    TICLCandidate(Charge q, const LorentzVector& p4) : LeafCandidate(q, p4) {}

    TICLCandidate() = default;

    TICLCandidate(const edm::Ptr<ticl::Trackster>& trackster)
        : LeafCandidate(), tracksters_({trackster}), time_(trackster->time()), timeError_(trackster->timeError()) {}

    TICLCandidate(const edm::Ptr<reco::Track> trackPtr, const edm::Ptr<ticl::Trackster>& tracksterPtr)
        : LeafCandidate() {
      if (trackPtr.isNull() and tracksterPtr.isNull())
        throw cms::Exception("NullPointerError")
            << "TICLCandidate constructor: at least one between track and trackster must be valid";

      if (tracksterPtr.isNonnull()) {
        tracksters_.push_back(tracksterPtr);
        auto const& trackster = tracksters_[0].get();
        setIdProbabilities(trackster->id_probabilities());
        if (trackPtr.isNonnull()) {
          trackPtrs_.push_back(trackPtr);
          auto const& tk = trackPtr.get();
          // PDG sign convention: a pi+ is +211, an e- is +11.
          setPdgId(trackster->isHadronic() ? 211 * tk->charge() : -11 * tk->charge());
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
        trackPtrs_.push_back(trackPtr);
        auto const& tk = trackPtr.get();
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

    inline const edm::Ptr<reco::Track> trackPtr(int index = 0) const {
      return trackPtrs_.empty() ? edm::Ptr<reco::Track>() : trackPtrs_[index];
    }
    inline const std::vector<edm::Ptr<reco::Track>> trackPtrs() const { return trackPtrs_; }
    void addTrackPtr(const edm::Ptr<reco::Track>& trackPtr) { trackPtrs_.push_back(trackPtr); }

    inline const edm::Ptr<reco::GsfTrack> gsftrackPtr(int index = 0) const {
      return gsfTrackPtrs_.empty() ? edm::Ptr<reco::GsfTrack>() : gsfTrackPtrs_[index];
    }
    inline const std::vector<edm::Ptr<reco::GsfTrack>>& gsfTrackPtrs() const { return gsfTrackPtrs_; }
    void addGsfTrackPtr(const edm::Ptr<reco::GsfTrack>& gsftrackPtr) { gsfTrackPtrs_.push_back(gsftrackPtr); }

    inline float rawEnergy() const { return rawEnergy_; }
    void setRawEnergy(float rawEnergy) { rawEnergy_ = rawEnergy; }

    inline const std::vector<edm::Ptr<ticl::Trackster>> tracksters() const { return tracksters_; };

    void setTracksters(const std::vector<edm::Ptr<ticl::Trackster>>& tracksters) { tracksters_ = tracksters; }
    // Adds a trackster and takes its time as the candidate time.
    void addTrackster(const edm::Ptr<ticl::Trackster>& trackster) {
      tracksters_.push_back(trackster);
      time_ = trackster->time();
      timeError_ = trackster->timeError();
    }
    // Adds a trackster and keeps the candidate time.
    void appendTrackster(const edm::Ptr<ticl::Trackster>& trackster) { tracksters_.push_back(trackster); }
    // convenience method to return the ID probability for a certain particle type
    inline float id_probability(ParticleType type) const {
      // probabilities are stored in the same order as defined in the ParticleType enum
      return decodeProbability(idProbabilities_[(int)type]);
    }

    inline std::array<float, 8> idProbabilities() const { return decodeProbabilities(idProbabilities_); }

    void zeroProbabilities() { idProbabilities_.fill(0); }

    void setIdProbabilities(const std::array<float, 8>& idProbs) { idProbabilities_ = encodeProbabilities(idProbs); }
    inline void setIdProbability(ParticleType type, float value) {
      idProbabilities_[int(type)] = encodeProbability(value);
    }

    // Probability of each candidate type from the interpretation stage, in the order of ParticleType (0 when not set).
    // A charged candidate holds the probabilities of the best hypothesis of each type of its track (electron, muon,
    // charged hadron): independent probabilities of competing hypotheses, not a distribution, and 0 for a type with no
    // hypothesis. A neutral candidate holds the species probabilities (photon, neutral pion, neutral hadron).
    inline std::array<float, 8> typeProbabilities() const { return decodeProbabilities(typeProbabilities_); }
    inline float typeProbability(ParticleType type) const { return decodeProbability(typeProbabilities_[int(type)]); }
    void setTypeProbabilities(const std::array<float, 8>& probabilities) {
      typeProbabilities_ = encodeProbabilities(probabilities);
    }

    // A probability is stored as a 16-bit integer, with 65535 for 1, rounded to the nearest step.
    // NaN and values below 0 are stored as 0, values above 1 as 65535.
    static constexpr float kProbabilityScale = 65535.f;
    static std::uint16_t encodeProbability(float p) {
      if (!(p > 0.f))
        return 0;
      if (p >= 1.f)
        return static_cast<std::uint16_t>(kProbabilityScale);
      return static_cast<std::uint16_t>(std::lround(static_cast<double>(p) * kProbabilityScale));
    }
    static float decodeProbability(std::uint16_t stored) { return stored / kProbabilityScale; }

  private:
    static std::array<std::uint16_t, 8> encodeProbabilities(const std::array<float, 8>& probabilities) {
      std::array<std::uint16_t, 8> stored;
      for (unsigned int i = 0; i < stored.size(); ++i)
        stored[i] = encodeProbability(probabilities[i]);
      return stored;
    }
    static std::array<float, 8> decodeProbabilities(const std::array<std::uint16_t, 8>& stored) {
      std::array<float, 8> probabilities;
      for (unsigned int i = 0; i < probabilities.size(); ++i)
        probabilities[i] = decodeProbability(stored[i]);
      return probabilities;
    }

    // vector of Ptr so Tracksters can come from different collections
    // and there can be derived classes
    std::vector<edm::Ptr<ticl::Trackster>> tracksters_;
    std::vector<edm::Ptr<reco::Track>> trackPtrs_;
    std::vector<edm::Ptr<reco::GsfTrack>> gsfTrackPtrs_;
    // Since it contains multiple tracksters, duplicate the probability interface
    std::array<std::uint16_t, 8> idProbabilities_{};
    std::array<std::uint16_t, 8> typeProbabilities_{};

    float time_ = 0.f;
    float timeError_ = -1.f;
    float MTDtime_ = 0.f;
    float MTDtimeError_ = -1.f;
    float rawEnergy_ = 0.f;
  };
}  // namespace io_v1
using TICLCandidate = io_v1::TICLCandidate;
#endif
