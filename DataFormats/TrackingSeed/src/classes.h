#ifndef TRACKINGSEED_CLASSES_H
#define TRACKINGSEED_CLASSES_H

#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  struct dictionary {
    edm::Wrapper<TrackingSeedCollection> trackingSeedCollectionWrapper;
  };
}

#endif // TRACKINGSEED_CLASSES_H
