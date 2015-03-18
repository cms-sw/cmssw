#ifndef DATAFORMATS_SiTrackerGSMatchedRecHit2DCollection_H
#define DATAFORMATS_SiTrackerGSMatchedRecHit2DCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include <vector>

typedef edm::RangeMap <unsigned, edm::OwnVector<SiTrackerGSMatchedRecHit2D> > SiTrackerGSMatchedRecHit2DCollection;
typedef edmNew::DetSetVector<SiTrackerGSMatchedRecHit2D> TrackerGSMatchedRecHitCollection;

#endif

