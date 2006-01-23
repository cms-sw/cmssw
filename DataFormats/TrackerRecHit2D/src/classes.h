#ifndef SISTRIPRECHIT_CLASSES_H
#define SISTRIPRECHIT_CLASSES_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<SiStripRecHit2DLocalPosCollection> siStripRecHit2DLocalPosCollectionWrapper;
    edm::Wrapper<SiStripRecHit2DMatchedLocalPosCollection> siStripRecHit2DmatchedLocalPosCollectionWrapper;
  }
}

#endif // SISTRIPRECHIT_CLASSES_H
