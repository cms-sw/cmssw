#ifndef ImpactParameter_JetProbabilityComputer_h
#define ImpactParameter_JetProbabilityComputer_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedJetProbabilityComputer.h"


typedef TemplatedJetProbabilityComputer<reco::TrackRefVector,reco::JTATagInfo> JetProbabilityComputer;

#endif // ImpactParameter_JetProbabilityComputer_h
