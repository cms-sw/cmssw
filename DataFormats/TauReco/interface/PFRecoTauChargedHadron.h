#ifndef DataFormats_TauReco_PFRecoTauChargedHadron_h
#define DataFormats_TauReco_PFRecoTauChargedHadron_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace reco 
{

namespace tau
{
  class PFRecoTauChargedHadronFromPFCandidatePlugin;
  class PFRecoTauChargedHadronFromTrackPlugin;
}

class PFRecoTauChargedHadron : public LeafCandidate 
{
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

  /// constructor from values
  PFRecoTauChargedHadron(Charge q, const LorentzVector& p4,
			 const Point& vtx = Point( 0, 0, 0 ),
			 int status = 0, bool integerCharge = true,
			 PFRecoTauChargedHadronAlgorithm algo = kUndefined);

  /// constructor from a Candidate
  PFRecoTauChargedHadron(const Candidate& c, PFRecoTauChargedHadronAlgorithm algo = kUndefined);
 
  /// destructor
  ~PFRecoTauChargedHadron();

  /// reference to "charged" PFCandidate (either charged PFCandidate or PFNeutralHadron)
  const PFCandidatePtr& getChargedPFCandidate() const;

  /// reference to reco::Track
  const TrackPtr& getTrack() const;

  /// references to additional neutral PFCandidates
  const std::vector<PFCandidatePtr>& getNeutralPFCandidates() const;  

  /// position at ECAL entrance
  const math::XYZPointF& positionAtECALEntrance() const;

  /// Algorithm that built this charged hadron
  PFRecoTauChargedHadronAlgorithm algo() const;

  /// Check whether a given algo produced this charged hadron
  bool algoIs(PFRecoTauChargedHadronAlgorithm algo) const;

  void print(std::ostream& stream = std::cout) const;
    
  friend class tau::PFRecoTauChargedHadronFromPFCandidatePlugin;
  friend class tau::PFRecoTauChargedHadronFromTrackPlugin;
  
 private:
  PFRecoTauChargedHadronAlgorithm algo_;

  PFCandidatePtr chargedPFCandidate_;
  TrackPtr track_;
  std::vector<PFCandidatePtr> neutralPFCandidates_;

  math::XYZPointF positionAtECALEntrance_;
};

std::ostream& operator<<(std::ostream& stream, const PFRecoTauChargedHadron& c);

}

#endif
