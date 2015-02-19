#ifndef ImpactParameter_NegativeTrackCountingComputer_h
#define ImpactParameter_NegativeTrackCountingComputer_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "RecoBTag/ImpactParameter/interface/TemplatedNegativeTrackCountingComputer.h"


typedef TemplatedNegativeTrackCountingComputer<reco::TrackRefVector,reco::JTATagInfo> NegativeTrackCountingComputer;

#endif // ImpactParameter_NegativeTrackCountingComputer_h