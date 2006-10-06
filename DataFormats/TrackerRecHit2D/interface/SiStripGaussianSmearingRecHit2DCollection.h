#ifndef DATAFORMATS_SISTRIPGAUSSIANSMEARINGRECHIT2DCOLLECTION_H
#define DATAFORMATS_SISTRIPGAUSSIANSMEARINGRECHIT2DCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripGaussianSmearingRecHit2D.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap <DetId, edm::OwnVector<SiStripGaussianSmearingRecHit2D> > SiStripGaussianSmearingRecHit2DCollection;


#endif

