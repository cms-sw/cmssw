// Authors: Felice Pantaleo, Marco Rovere
// Emails: felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 06/2019

#ifndef DataFormats_HGCalReco_TICLSeedingRegion_h
#define DataFormats_HGCalReco_TICLSeedingRegion_h

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Provenance/interface/ProductID.h"

struct TICLSeedingRegion {
  GlobalPoint origin;
  GlobalVector directionAtOrigin;

  // zSide can be either 0(neg) or 1(pos)
  int zSide;
  // the index in the seeding collection
  // with index = -1 indicating a global seeding region
  int index;
  // collectionID = 0 used for global seeding collection
  edm::ProductID collectionID;

  TICLSeedingRegion() {}

  TICLSeedingRegion(GlobalPoint o, GlobalVector d, int zS, int idx, edm::ProductID id) {
    origin = o;
    directionAtOrigin = d;
    zSide = zS;
    index = idx;
    collectionID = id;
  }
};

#endif
