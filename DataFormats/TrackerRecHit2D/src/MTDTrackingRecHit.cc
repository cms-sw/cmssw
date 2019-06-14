#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

void MTDTrackingRecHit::getKfComponents(KfComponentsHolder& holder) const { getKfComponents2D(holder); }
