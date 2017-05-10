#ifndef RecoTauTag_RecoTau_pfRecoTauChargedHadronAuxFunctions_h
#define RecoTauTag_RecoTau_pfRecoTauChargedHadronAuxFunctions_h

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"

namespace reco { namespace tau {

    void setChargedHadronP4(reco::PFRecoTauChargedHadron& chargedHadron, double scaleFactor_neutralPFCands = 1.0);
    reco::Candidate::LorentzVector compChargedHadronP4fromPxPyPz(double, double, double);
    reco::Candidate::LorentzVector compChargedHadronP4fromPThetaPhi(double, double, double);

}} // end namespace reco::tau

#endif
