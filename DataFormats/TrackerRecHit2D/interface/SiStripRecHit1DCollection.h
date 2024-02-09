#ifndef DATAFORMATS_SISTRIPRECHIT1DCOLLECTION_H
#define DATAFORMATS_SISTRIPRECHIT1DCOLLECTION_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

typedef edmNew::DetSetVector<SiStripRecHit1D> SiStripRecHit1DCollection;
typedef SiStripRecHit1DCollection SiStripRecHit1DCollectionNew;

#endif
