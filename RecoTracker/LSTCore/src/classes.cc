#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"

#ifndef LST_STANDALONE
SET_PORTABLEHOSTCOLLECTION_READ_RULES(lst::LSTInputHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(lst::TrackCandidatesBaseHostCollection);
#endif
