#define TRAJECTORYSEED_CLASSES_H
#ifndef TRAJECTORYSEED_CLASSES_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    TrackCharge tc;
    TrajectorySeed ts;
    LocalTrajectoryParameters ppp;
    edm::OwnVector<TrackingRecHit> rhCollection;
    TrajectorySeedCollection coll;
    edm::Wrapper<TrajectorySeedCollection> TrajectorySeedCollectionWrapper;
  }
}

#endif
