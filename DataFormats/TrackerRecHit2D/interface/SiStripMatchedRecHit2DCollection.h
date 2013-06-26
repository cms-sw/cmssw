#ifndef DATAFORMATS_SISTRIPMATCHEDRECHIT2DCOLLECTION_H
#define DATAFORMATS_SISTRIPMATCHEDRECHIT2DCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <vector>

typedef  edm::RangeMap<DetId, edm::OwnVector<SiStripMatchedRecHit2D> > SiStripMatchedRecHit2DCollectionOld;


#include "DataFormats/Common/interface/DetSetVectorNew.h"
typedef edmNew::DetSetVector<SiStripMatchedRecHit2D> SiStripMatchedRecHit2DCollection;
typedef SiStripMatchedRecHit2DCollection SiStripMatchedRecHit2DCollectionNew;


#endif
