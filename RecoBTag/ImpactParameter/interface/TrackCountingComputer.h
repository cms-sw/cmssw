#ifndef ImpactParameter_TrackCountingComputer_h
#define ImpactParameter_TrackCountingComputer_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedTrackCountingComputer.h"


typedef TemplatedTrackCountingComputer<reco::TrackRefVector,reco::JTATagInfo> TrackCountingComputer;

#endif // ImpactParameter_TrackCountingComputer_h
