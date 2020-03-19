#ifndef FASTSIMDATAFORMATS_FastTrackerClusterCollection_H
#define FASTSIMDATAFORMATS_FastTrackerClusterCollection_H

#include "FastSimDataFormats/External/interface/FastTrackerCluster.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap<unsigned, edm::OwnVector<FastTrackerCluster> > FastTrackerClusterCollection;

#endif
