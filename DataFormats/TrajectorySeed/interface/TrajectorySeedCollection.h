#ifndef DATAFORMATS_TRAJECTORYSEED_TRAJECTORYSEEDCOLLECTION_h
#define DATAFORMATS_TRAJECTORYSEED_TRAJECTORYSEEDCOLLECTION_h

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/NewPolicy.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

typedef edm::OwnVector<TrajectorySeed, edm::NewPolicy<TrajectorySeed> > TrajectorySeedCollection;

#endif
