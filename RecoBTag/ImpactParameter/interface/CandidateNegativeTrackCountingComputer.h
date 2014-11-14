#ifndef ImpactParameter_CandNegativeTrackCountingComputer_h
#define ImpactParameter_CandNegativeTrackCountingComputer_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedNegativeTrackCountingComputer.h"


typedef TemplatedNegativeTrackCountingComputer<std::vector<reco::CandidatePtr>,reco::JetTagInfo> CandidateNegativeTrackCountingComputer;

#endif // ImpactParameter_CandNegativeTrackCountingComputer_h