#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
using namespace std;
using namespace edm;
using namespace reco;

PFTauTagInfo* PFTauTagInfo::clone() const { return new PFTauTagInfo(*this); }

std::vector<reco::CandidatePtr> PFTauTagInfo::PFCands() const {
  std::vector<reco::CandidatePtr> thePFCands;
  for (std::vector<reco::CandidatePtr>::const_iterator iPFCand = PFChargedHadrCands_.begin();
       iPFCand != PFChargedHadrCands_.end();
       iPFCand++)
    thePFCands.push_back(*iPFCand);
  for (std::vector<reco::CandidatePtr>::const_iterator iPFCand = PFNeutrHadrCands_.begin();
       iPFCand != PFNeutrHadrCands_.end();
       iPFCand++)
    thePFCands.push_back(*iPFCand);
  for (std::vector<reco::CandidatePtr>::const_iterator iPFCand = PFGammaCands_.begin(); iPFCand != PFGammaCands_.end();
       iPFCand++)
    thePFCands.push_back(*iPFCand);
  return thePFCands;
}
const std::vector<reco::CandidatePtr>& PFTauTagInfo::PFChargedHadrCands() const { return PFChargedHadrCands_; }
void PFTauTagInfo::setPFChargedHadrCands(const std::vector<reco::CandidatePtr>& x) { PFChargedHadrCands_ = x; }
const std::vector<reco::CandidatePtr>& PFTauTagInfo::PFNeutrHadrCands() const { return PFNeutrHadrCands_; }
void PFTauTagInfo::setPFNeutrHadrCands(const std::vector<reco::CandidatePtr>& x) { PFNeutrHadrCands_ = x; }
const std::vector<reco::CandidatePtr>& PFTauTagInfo::PFGammaCands() const { return PFGammaCands_; }
void PFTauTagInfo::setPFGammaCands(const std::vector<reco::CandidatePtr>& x) { PFGammaCands_ = x; }

const JetBaseRef& PFTauTagInfo::pfjetRef() const { return PFJetRef_; }
void PFTauTagInfo::setpfjetRef(const JetBaseRef x) { PFJetRef_ = x; }
