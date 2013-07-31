#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

namespace reco {

PFRecoTauChargedHadron::PFRecoTauChargedHadron()
  : CompositePtrCandidate(),
    algo_(kUndefined)
{}

PFRecoTauChargedHadron::PFRecoTauChargedHadron(PFRecoTauChargedHadronAlgorithm algo, Charge q)
  : CompositePtrCandidate(), 
    algo_(algo) 
{ 
  if ( q > 0. ) this->setPdgId(+211); 
  else if ( q < 0. ) this->setPdgId(-211); 
}

PFRecoTauChargedHadron::PFRecoTauChargedHadron(Charge q, const LorentzVector& p4,
					       const Point& vtx,
					       int status, bool integerCharge,
					       PFRecoTauChargedHadronAlgorithm algo)
  : CompositePtrCandidate(q, p4, vtx, 211, status, integerCharge), 
    algo_(algo) 
{ 
  if ( q > 0. ) this->setPdgId(+211); 
  else if ( q < 0. ) this->setPdgId(-211);   
}
    
PFRecoTauChargedHadron::PFRecoTauChargedHadron(const Candidate& c, PFRecoTauChargedHadronAlgorithm algo)
  : CompositePtrCandidate(c),
    algo_(algo) 
{ 
  if ( c.charge() > 0. ) this->setPdgId(+211); 
  else if ( c.charge() < 0. ) this->setPdgId(-211); 
}
  
PFRecoTauChargedHadron::~PFRecoTauChargedHadron()
{}

const PFCandidatePtr& PFRecoTauChargedHadron::getChargedPFCandidate() const
{
  return chargedPFCandidate_;
}

const PFRecoTauChargedHadron::TrackPtr& PFRecoTauChargedHadron::getTrack() const
{
  return track_;
}

const std::vector<PFCandidatePtr>& PFRecoTauChargedHadron::getNeutralPFCandidates() const
{
  return neutralPFCandidates_;
}

const math::XYZPointF& PFRecoTauChargedHadron::positionAtECALEntrance() const 
{
  return positionAtECALEntrance_;
}

PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm PFRecoTauChargedHadron::algo() const 
{
  return algo_;
}

bool PFRecoTauChargedHadron::algoIs(PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm algo) const 
{
  return (algo_ == algo);
}

namespace
{
  std::string getPFCandidateType(reco::PFCandidate::ParticleType pfCandidateType)
  {
    if      ( pfCandidateType == reco::PFCandidate::X         ) return "undefined";
    else if ( pfCandidateType == reco::PFCandidate::h         ) return "PFChargedHadron";
    else if ( pfCandidateType == reco::PFCandidate::e         ) return "PFElectron";
    else if ( pfCandidateType == reco::PFCandidate::mu        ) return "PFMuon";
    else if ( pfCandidateType == reco::PFCandidate::gamma     ) return "PFGamma";
    else if ( pfCandidateType == reco::PFCandidate::h0        ) return "PFNeutralHadron";
    else if ( pfCandidateType == reco::PFCandidate::h_HF      ) return "HF_had";
    else if ( pfCandidateType == reco::PFCandidate::egamma_HF ) return "HF_em";
    else assert(0);
  }
}

void PFRecoTauChargedHadron::print(std::ostream& stream) const 
{
  stream << " Pt = " << this->pt() << ", eta = " << this->eta() << ", phi = " << this->phi() << " (mass = " << this->mass() << ")" << std::endl;
  stream << " charge = " << this->charge() << " (pdgId = " << this->pdgId() << ")" << std::endl;
  stream << "charged PFCandidate";
  if ( chargedPFCandidate_.isNonnull() ) {
    stream << " (" << chargedPFCandidate_.id() << ":" << chargedPFCandidate_.key() << "):"
	   << " Pt = " << chargedPFCandidate_->pt() << ", eta = " << chargedPFCandidate_->eta() << ", phi = " << chargedPFCandidate_->phi() 
	   << " (type = " << getPFCandidateType(chargedPFCandidate_->particleId()) << ")" << std::endl;
  } else {
    stream << ": N/A" << std::endl;
  }
  stream << "reco::Track: ";
  if ( track_.isNonnull() ) {
    stream << "Pt = " << track_->pt() << " +/- " << track_->ptError() << ", eta = " << track_->eta() << ", phi = " << track_->phi() << std::endl;
  } else {
    stream << "N/A" << std::endl;
  }
  stream << "neutral PFCandidates:";
  if ( neutralPFCandidates_.size() >= 1 ) {
    stream << std::endl;
    int idx = 0;
    for ( std::vector<PFCandidatePtr>::const_iterator neutralPFCandidate = neutralPFCandidates_.begin();
	  neutralPFCandidate != neutralPFCandidates_.end(); ++neutralPFCandidate ) {
      stream << " #" << idx << " (" << neutralPFCandidate->id() << ":" << neutralPFCandidate->key() << "):"
	     << " Pt = " << (*neutralPFCandidate)->pt() << ", eta = " << (*neutralPFCandidate)->eta() << ", phi = " << (*neutralPFCandidate)->phi() 
	     << " (type = " << getPFCandidateType((*neutralPFCandidate)->particleId()) << ")" << std::endl;
      ++idx;
    }
  } else {
    stream << " ";
    stream << "N/A" << std::endl;
  }
  stream << "position@ECAL entrance: x = " << this->positionAtECALEntrance().x() << ", y = " << this->positionAtECALEntrance().y() << ", z = " << this->positionAtECALEntrance().z() 
	 << " (eta = " << this->positionAtECALEntrance().eta() << ", phi = " << this->positionAtECALEntrance().phi() << ")" << std::endl;
  std::string algo_string = "undefined";
  if      ( algo_ == kChargedPFCandidate ) algo_string = "chargedPFCandidate";
  else if ( algo_ == kTrack              ) algo_string = "Track";
  else if ( algo_ == kPFNeutralHadron    ) algo_string = "PFNeutralHadron";
  stream << "algo = " << algo_string << std::endl;
}

std::ostream& operator<<(std::ostream& stream, const reco::PFRecoTauChargedHadron& c)
{
  c.print(stream);
  return stream;
}
}
