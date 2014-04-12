#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
using namespace std;
using namespace edm;
using namespace reco;

PFTauTagInfo* PFTauTagInfo::clone()const{return new PFTauTagInfo(*this);}

std::vector<reco::PFCandidatePtr> PFTauTagInfo::PFCands()const{
  std::vector<reco::PFCandidatePtr> thePFCands;
  for (std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  for (std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=PFNeutrHadrCands_.begin();iPFCand!=PFNeutrHadrCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  for (std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=PFGammaCands_.begin();iPFCand!=PFGammaCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  return thePFCands;
}
const std::vector<reco::PFCandidatePtr>& PFTauTagInfo::PFChargedHadrCands() const {return PFChargedHadrCands_;}
void  PFTauTagInfo::setPFChargedHadrCands(const std::vector<reco::PFCandidatePtr>& x){PFChargedHadrCands_=x;}
const std::vector<reco::PFCandidatePtr>& PFTauTagInfo::PFNeutrHadrCands() const {return PFNeutrHadrCands_;}
void  PFTauTagInfo::setPFNeutrHadrCands(const std::vector<reco::PFCandidatePtr>& x){PFNeutrHadrCands_=x;}
const std::vector<reco::PFCandidatePtr>& PFTauTagInfo::PFGammaCands() const {return PFGammaCands_;}
void  PFTauTagInfo::setPFGammaCands(const std::vector<reco::PFCandidatePtr>& x){PFGammaCands_=x;}

const PFJetRef& PFTauTagInfo::pfjetRef()const{return PFJetRef_;}
void PFTauTagInfo::setpfjetRef(const PFJetRef x){PFJetRef_=x;}
