#ifndef DataFormats_SiPixelRecHitCollection_H
#define DataFormats_SiPixelRecHitCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <vector>

typedef  edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHit> > SiPixelRecHitCollectionOld;

// new collection(for some far indetermined future)
#include "DataFormats/Common/interface/DetSetVectorNew.h"
typedef edmNew::DetSetVector<SiPixelRecHit> SiPixelRecHitCollection;
typedef SiPixelRecHitCollection             SiPixelRecHitCollectionNew;

#endif


