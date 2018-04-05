#ifndef FastTrackerRecHitCollection_H
#define FastTrackerRecHitCollection_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitFwd.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ref.h"

typedef edm::OwnVector<FastTrackerRecHit> FastTrackerRecHitCollection;
typedef edm::Ref<FastTrackerRecHitCollection> FastTrackerRecHitRef;
typedef std::vector<FastTrackerRecHitRef> FastTrackerRecHitRefCollection;
typedef std::vector<FastTrackerRecHitRef> FastTrackerRecHitCombination;
typedef std::vector<FastTrackerRecHitCombination> FastTrackerRecHitCombinationCollection;
typedef edm::Ref<FastTrackerRecHitCombinationCollection> FastTrackerRecHitCombinationRef;

#endif
