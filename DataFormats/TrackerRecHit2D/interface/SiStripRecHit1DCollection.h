#ifndef DATAFORMATS_SISTRIPRECHIT1DCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT1DCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap<DetId, edm::OwnVector<SiStripRecHit1D> > SiStripRecHit1DCollectionOld;

// new collection(for some far indetermined future)
#include "DataFormats/Common/interface/DetSetVectorNew.h"
typedef edmNew::DetSetVector<SiStripRecHit1D> SiStripRecHit1DCollection;
typedef SiStripRecHit1DCollection SiStripRecHit1DCollectionNew;

#endif
