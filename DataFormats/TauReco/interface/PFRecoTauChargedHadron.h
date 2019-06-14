#ifndef DataFormats_TauReco_PFRecoTauChargedHadron_h
#define DataFormats_TauReco_PFRecoTauChargedHadron_h

#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace reco {
  namespace tau {
    template <class TrackClass>
    class PFRecoTauChargedHadronFromGenericTrackPlugin;
    class PFRecoTauChargedHadronFromPFCandidatePlugin;
    class RecoTauConstructor;
    class PFRecoTauEnergyAlgorithmPlugin;
  }  // namespace tau
}  // namespace reco
class PFRecoTauChargedHadronProducer;

namespace reco {

  class PFRecoTauChargedHadron : public CompositePtrCandidate {
  public:
    typedef edm::Ptr<Track> TrackPtr;

    enum PFRecoTauChargedHadronAlgorithm {
      // Algorithm where each photon becomes a pi zero
      kUndefined = 0,
      kChargedPFCandidate = 1,
      kTrack = 2,
      kPFNeutralHadron = 3
    };

    PFRecoTauChargedHadron();
    PFRecoTauChargedHadron(PFRecoTauChargedHadronAlgorithm algo, Charge q);

    /// constructor from a Candidate
    PFRecoTauChargedHadron(const Candidate& c, PFRecoTauChargedHadronAlgorithm algo = kUndefined);

    /// constructor from values
    PFRecoTauChargedHadron(Charge q,
                           const LorentzVector& p4,
                           const Point& vtx = Point(0, 0, 0),
                           int status = 0,
                           bool integerCharge = true,
                           PFRecoTauChargedHadronAlgorithm algo = kUndefined);

    /// destructor
    ~PFRecoTauChargedHadron() override;

    /// reference to "charged" PFCandidate (either charged PFCandidate or PFNeutralHadron)
    const CandidatePtr& getChargedPFCandidate() const;

    /// reference to reco::Track
    const TrackPtr& getTrack() const;

    /// reference to "lostTrack Candidate" when chadron built with tracks stored as pat::PackedCandidates
    const CandidatePtr& getLostTrackCandidate() const;

    /// references to additional neutral PFCandidates
    const std::vector<CandidatePtr>& getNeutralPFCandidates() const;

    /// position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const;

    /// Algorithm that built this charged hadron
    PFRecoTauChargedHadronAlgorithm algo() const;

    /// Check whether a given algo produced this charged hadron
    bool algoIs(PFRecoTauChargedHadronAlgorithm algo) const;

    void print(std::ostream& stream = std::cout) const;

  private:
    friend class tau::PFRecoTauChargedHadronFromPFCandidatePlugin;
    template <class TrackClass>
    friend class tau::PFRecoTauChargedHadronFromGenericTrackPlugin;
    friend class tau::RecoTauConstructor;
    friend class tau::PFRecoTauEnergyAlgorithmPlugin;
    friend class ::PFRecoTauChargedHadronProducer;

    PFRecoTauChargedHadronAlgorithm algo_;

    CandidatePtr chargedPFCandidate_;
    CandidatePtr lostTrackCandidate_;
    TrackPtr track_;
    std::vector<CandidatePtr> neutralPFCandidates_;

    math::XYZPointF positionAtECALEntrance_;
  };

  std::ostream& operator<<(std::ostream& stream, const PFRecoTauChargedHadron& c);

}  // end namespace reco

#endif
