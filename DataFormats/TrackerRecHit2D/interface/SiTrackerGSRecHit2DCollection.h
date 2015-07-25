#ifndef DATAFORMATS_SiTrackerGSRecHit2DCollection_H
#define DATAFORMATS_SiTrackerGSRecHit2DCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include <vector>

typedef edm::RangeMap <unsigned, edm::OwnVector<SiTrackerGSRecHit2D> > SiTrackerGSRecHit2DCollection;
typedef edmNew::DetSetVector<SiTrackerGSRecHit2D> TrackerGSRecHitCollection;

#endif

