#ifndef ImpactParameter_CandTrackCountingComputer_h
#define ImpactParameter_CandTrackCountingComputer_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedTrackCountingComputer.h"


typedef TemplatedTrackCountingComputer<std::vector<reco::CandidatePtr>,reco::JetTagInfo> CandidateTrackCountingComputer;

#endif // ImpactParameter_CandTrackCountingComputer_h
