#ifndef ImpactParameter_JetBProbabilityComputer_h
#define ImpactParameter_JetBProbabilityComputer_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedJetBProbabilityComputer.h"


typedef TemplatedJetBProbabilityComputer<reco::TrackRefVector,reco::JTATagInfo> JetBProbabilityComputer;

#endif // ImpactParameter_JetBProbabilityComputer_h
