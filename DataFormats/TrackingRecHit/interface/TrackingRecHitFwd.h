#ifndef TrackingRecHit_TrackingRecHitFwd_h
#define TrackingRecHit_TrackingRecHitFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Utilities/interface/Range.h"

class TrackingRecHit;
/// collection of TrackingRecHits
typedef  edm::OwnVector<TrackingRecHit> TrackingRecHitCollection;
/// persistent reference to a TrackingRecHit
typedef edm::Ref<TrackingRecHitCollection> TrackingRecHitRef;
/// persistent reference to a TrackingRecHit collection
typedef edm::RefProd<TrackingRecHitCollection> TrackingRecHitRefProd;
/// vector of reference to TrackingRecHit in the same collection
typedef edm::RefVector<TrackingRecHitCollection> TrackingRecHitRefVector;
/// iterator over a vector of reference to TrackingRecHit in the same collection
typedef TrackingRecHitCollection::base::const_iterator trackingRecHit_iterator;
/// Range class to enable range-based loops for a tracks RecHits
using TrackingRecHitRange = edm::Range<trackingRecHit_iterator>;

#endif
