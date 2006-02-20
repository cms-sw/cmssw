#ifndef DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT2DMATCHEDLOCALPOSCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <vector>

typedef  edm::RangeMap<DetId, edm::OwnVector<SiStripRecHit2DMatchedLocalPos, edm::ClonePolicy<SiStripRecHit2DMatchedLocalPos> >, edm::ClonePolicy<SiStripRecHit2DMatchedLocalPos> > SiStripRecHit2DMatchedLocalPosCollection;

#endif
