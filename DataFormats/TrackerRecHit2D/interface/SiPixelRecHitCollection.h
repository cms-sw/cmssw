#ifndef DataFormats_SiPixelRecHitCollection_H
#define DataFormats_SiPixelRecHitCollection_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
typedef edmNew::DetSetVector<SiPixelRecHit> SiPixelRecHitCollection;
typedef SiPixelRecHitCollection             SiPixelRecHitCollectionNew;

#endif


