#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace reco {

size_t RecoTauPiZero::numberOfGammas() const
{
  size_t nGammas = 0;
  size_t nDaughters = numberOfDaughters();
  for(size_t i = 0; i < nDaughters; ++i) {
    if(daughter(i)->pdgId() == 22) ++nGammas;
  }
  return nGammas;
}

size_t RecoTauPiZero::numberOfElectrons() const
{
  size_t nElectrons = 0;
  size_t nDaughters = numberOfDaughters();
  for(size_t i = 0; i < nDaughters; ++i) {
    if(std::abs(daughter(i)->pdgId()) == 11) ++nElectrons;
  }
  return nElectrons;
}

double RecoTauPiZero::maxDeltaPhi() const
{
  double maxDPhi = 0;
  size_t nDaughters = numberOfDaughters();
  for(size_t i = 0; i < nDaughters; ++i) {
    double dPhi = std::fabs(deltaPhi(*this, *daughter(i)));
    if(dPhi > maxDPhi)
      maxDPhi = dPhi;
  }
  return maxDPhi;
}

double RecoTauPiZero::maxDeltaEta() const
{
  double maxDEta = 0;
  size_t nDaughters = numberOfDaughters();
  for(size_t i = 0; i < nDaughters; ++i) {
    double dEta = std::fabs(eta() - daughter(i)->eta());
    if(dEta > maxDEta)
      maxDEta = dEta;
  }
  return maxDEta;
}

RecoTauPiZero::PiZeroAlgorithm RecoTauPiZero::algo() const {
  return algoName_;
}

bool RecoTauPiZero::algoIs(RecoTauPiZero::PiZeroAlgorithm algo) const {
  return (algoName_ == algo);
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

void RecoTauPiZero::print(std::ostream& stream) const 
{
  std::cout << "Pt = " << this->pt() << ", eta = " << this->eta() << ", phi = " << this->phi() << std::endl;
  size_t numDaughters = this->numberOfDaughters();
  for ( size_t iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
    const reco::PFCandidate* daughter = dynamic_cast<const reco::PFCandidate*>(this->daughterPtr(iDaughter).get());
    std::cout << " daughter #" << iDaughter << " (" << getPFCandidateType(daughter->particleId()) << "):"
	      << " Pt = " << daughter->pt() << ", eta = " << daughter->eta() << ", phi = " << daughter->phi() << std::endl;
  }
}

std::ostream& operator<<(std::ostream& out, const reco::RecoTauPiZero& piZero)
{
  if(!out) return out;
  piZero.print(out);
  return out;
}
}
