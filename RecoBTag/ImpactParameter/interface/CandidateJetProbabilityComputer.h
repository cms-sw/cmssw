#ifndef ImpactParameter_CandidateJetProbabilityComputer_h
#define ImpactParameter_CandidateJetProbabilityComputer_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedJetProbabilityComputer.h"


typedef TemplatedJetProbabilityComputer<std::vector<reco::CandidatePtr>,reco::JetTagInfo> CandidateJetProbabilityComputer;

#endif // ImpactParameter_CandidateJetProbabilityComputer_h
