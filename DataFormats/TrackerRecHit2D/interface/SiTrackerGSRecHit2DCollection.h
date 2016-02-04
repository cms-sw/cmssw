#ifndef DATAFORMATS_SiTrackerGSRecHit2DCollection_H
#define DATAFORMATS_SiTrackerGSRecHit2DCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap <DetId, edm::OwnVector<SiTrackerGSRecHit2D> > SiTrackerFullGSRecHit2DCollection;
typedef edm::RangeMap <unsigned, edm::OwnVector<SiTrackerGSRecHit2D> > SiTrackerGSRecHit2DCollection;

#endif

