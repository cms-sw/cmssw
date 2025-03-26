#ifndef DataFormats_TrackerRecHit2D_SiPixelRecHitFwd_h
#define DataFormats_TrackerRecHit2D_SiPixelRecHitFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

// persistent reference to a SiPixelRecHit in a SiPixelRecHitCollection
typedef edm::Ref<SiPixelRecHitCollection, SiPixelRecHit> SiPixelRecHitRef;
// persistent vector of references to SiPixelRecHits in a SiPixelRecHitCollection
typedef edm::RefVector<SiPixelRecHitCollection, SiPixelRecHit> SiPixelRecHitRefVector;

#endif