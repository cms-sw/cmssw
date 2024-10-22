#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

namespace reco {

  PFRecoTauChargedHadron::PFRecoTauChargedHadron() : CompositePtrCandidate(), algo_(kUndefined) {}

  PFRecoTauChargedHadron::PFRecoTauChargedHadron(PFRecoTauChargedHadronAlgorithm algo, Charge q)
      : CompositePtrCandidate(), algo_(algo) {
    if (q > 0.)
      this->setPdgId(+211);
    else if (q < 0.)
      this->setPdgId(-211);
  }

  PFRecoTauChargedHadron::PFRecoTauChargedHadron(Charge q,
                                                 const LorentzVector& p4,
                                                 const Point& vtx,
                                                 int status,
                                                 bool integerCharge,
                                                 PFRecoTauChargedHadronAlgorithm algo)
      : CompositePtrCandidate(q, p4, vtx, 211, status, integerCharge), algo_(algo) {
    if (q > 0.)
      this->setPdgId(+211);
    else if (q < 0.)
      this->setPdgId(-211);
  }

  PFRecoTauChargedHadron::PFRecoTauChargedHadron(const Candidate& c, PFRecoTauChargedHadronAlgorithm algo)
      : CompositePtrCandidate(c), algo_(algo) {
    if (c.charge() > 0.)
      this->setPdgId(+211);
    else if (c.charge() < 0.)
      this->setPdgId(-211);
  }

  PFRecoTauChargedHadron::~PFRecoTauChargedHadron() {}

  const CandidatePtr& PFRecoTauChargedHadron::getChargedPFCandidate() const { return chargedPFCandidate_; }

  const PFRecoTauChargedHadron::TrackPtr& PFRecoTauChargedHadron::getTrack() const { return track_; }

  const CandidatePtr& PFRecoTauChargedHadron::getLostTrackCandidate() const { return lostTrackCandidate_; }

  const std::vector<CandidatePtr>& PFRecoTauChargedHadron::getNeutralPFCandidates() const {
    return neutralPFCandidates_;
  }

  const math::XYZPointF& PFRecoTauChargedHadron::positionAtECALEntrance() const { return positionAtECALEntrance_; }

  PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm PFRecoTauChargedHadron::algo() const { return algo_; }

  bool PFRecoTauChargedHadron::algoIs(PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm algo) const {
    return (algo_ == algo);
  }

  void PFRecoTauChargedHadron::print(std::ostream& stream) const {
    stream << " Pt = " << this->pt() << ", eta = " << this->eta() << ", phi = " << this->phi()
           << " (mass = " << this->mass() << ")" << std::endl;
    stream << " charge = " << this->charge() << " (pdgId = " << this->pdgId() << ")" << std::endl;
    stream << "charged PFCandidate";
    if (chargedPFCandidate_.isNonnull()) {
      stream << " (" << chargedPFCandidate_.id() << ":" << chargedPFCandidate_.key() << "):"
             << " Pt = " << chargedPFCandidate_->pt() << ", eta = " << chargedPFCandidate_->eta()
             << ", phi = " << chargedPFCandidate_->phi() << " (pdgId = " << chargedPFCandidate_->pdgId() << ")"
             << std::endl;
    } else {
      stream << ": N/A" << std::endl;
    }
    stream << "reco::Track: ";
    if (track_.isNonnull()) {
      stream << "Pt = " << track_->pt() << " +/- " << track_->ptError() << ", eta = " << track_->eta()
             << ", phi = " << track_->phi() << std::endl;
    } else if (lostTrackCandidate_.isNonnull()) {
      stream << "(lostTrackCandidate: " << lostTrackCandidate_.id() << ":" << lostTrackCandidate_.key() << "):"
             << " Pt = " << lostTrackCandidate_->pt() << ", eta = " << lostTrackCandidate_->eta()
             << ", phi = " << lostTrackCandidate_->phi() << std::endl;
    } else {
      stream << "N/A" << std::endl;
    }
    stream << "neutral PFCandidates:";
    if (!neutralPFCandidates_.empty()) {
      stream << std::endl;
      int idx = 0;
      for (std::vector<CandidatePtr>::const_iterator neutralPFCandidate = neutralPFCandidates_.begin();
           neutralPFCandidate != neutralPFCandidates_.end();
           ++neutralPFCandidate) {
        stream << " #" << idx << " (" << neutralPFCandidate->id() << ":" << neutralPFCandidate->key() << "):"
               << " Pt = " << (*neutralPFCandidate)->pt() << ", eta = " << (*neutralPFCandidate)->eta()
               << ", phi = " << (*neutralPFCandidate)->phi() << " (pdgId = " << (*neutralPFCandidate)->pdgId() << ")"
               << std::endl;
        ++idx;
      }
    } else {
      stream << " ";
      stream << "N/A" << std::endl;
    }
    stream << "position@ECAL entrance: x = " << this->positionAtECALEntrance().x()
           << ", y = " << this->positionAtECALEntrance().y() << ", z = " << this->positionAtECALEntrance().z()
           << " (eta = " << this->positionAtECALEntrance().eta() << ", phi = " << this->positionAtECALEntrance().phi()
           << ")" << std::endl;
    std::string algo_string = "undefined";
    if (algo_ == kChargedPFCandidate)
      algo_string = "chargedPFCandidate";
    else if (algo_ == kTrack)
      algo_string = "Track";
    else if (algo_ == kPFNeutralHadron)
      algo_string = "PFNeutralHadron";
    stream << "algo = " << algo_string << std::endl;
  }

  std::ostream& operator<<(std::ostream& stream, const reco::PFRecoTauChargedHadron& c) {
    c.print(stream);
    return stream;
  }
}  // namespace reco
