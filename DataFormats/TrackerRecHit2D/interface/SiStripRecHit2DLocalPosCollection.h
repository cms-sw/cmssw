#ifndef DATAFORMATS_SISTRIPRECHIT2DLOCALPOSCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT2DLOCALPOSCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap <DetId, edm::OwnVector<SiStripRecHit2DLocalPos,edm::ClonePolicy<SiStripRecHit2DLocalPos> >, edm::ClonePolicy<SiStripRecHit2DLocalPos> > SiStripRecHit2DLocalPosCollection;


#endif

