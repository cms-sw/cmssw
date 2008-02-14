#ifndef DATAFORMATS_SiTrackerGSMatchedRecHit2DCollection_H
#define DATAFORMATS_SiTrackerGSMatchedRecHit2DCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap <unsigned, edm::OwnVector<SiTrackerGSMatchedRecHit2D> > SiTrackerGSMatchedRecHit2DCollection;

#endif

