#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
using namespace std;
using namespace edm;
using namespace reco;

PFTauTagInfo* PFTauTagInfo::clone()const{return new PFTauTagInfo(*this);}

PFCandidateRefVector PFTauTagInfo::PFCands()const{
  PFCandidateRefVector thePFCands;
  for (PFCandidateRefVector::const_iterator iPFCand=PFChargedHadrCands_.begin();iPFCand!=PFChargedHadrCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  for (PFCandidateRefVector::const_iterator iPFCand=PFNeutrHadrCands_.begin();iPFCand!=PFNeutrHadrCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  for (PFCandidateRefVector::const_iterator iPFCand=PFGammaCands_.begin();iPFCand!=PFGammaCands_.end();iPFCand++) thePFCands.push_back(*iPFCand);
  return thePFCands;
}
const PFCandidateRefVector& PFTauTagInfo::PFChargedHadrCands() const {return PFChargedHadrCands_;}
void  PFTauTagInfo::setPFChargedHadrCands(const PFCandidateRefVector x){PFChargedHadrCands_=x;}
const PFCandidateRefVector& PFTauTagInfo::PFNeutrHadrCands() const {return PFNeutrHadrCands_;}
void  PFTauTagInfo::setPFNeutrHadrCands(const PFCandidateRefVector x){PFNeutrHadrCands_=x;}
const PFCandidateRefVector& PFTauTagInfo::PFGammaCands() const {return PFGammaCands_;}
void  PFTauTagInfo::setPFGammaCands(const PFCandidateRefVector x){PFGammaCands_=x;}

const PFJetRef& PFTauTagInfo::pfjetRef()const{return PFJetRef_;}
void PFTauTagInfo::setpfjetRef(const PFJetRef x){PFJetRef_=x;}
