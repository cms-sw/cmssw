#ifndef ImpactParameter_CandidateJetBProbabilityComputer_h
#define ImpactParameter_CandidateJetBProbabilityComputer_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedJetBProbabilityComputer.h"


typedef TemplatedJetBProbabilityComputer<std::vector<reco::CandidatePtr>,reco::JetTagInfo> CandidateJetBProbabilityComputer;

#endif // ImpactParameter_CandidateJetBProbabilityComputer_h
