// Authors: Felice Pantaleo, Marco Rovere 
// Emails: felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 06/2019

#ifndef RecoHGCal_TICL_TICLSeedingRegion_h
#define RecoHGCal_TICL_TICLSeedingRegion_h

#include "DataFormats/TICL/interface/Common.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

namespace ticl {
  struct TICLSeedingRegion {
    GlobalPoint origin;
    GlobalVector directionAtOrigin;

// zSide can be either 0 or 1
    int zSide;
// the index in the seeding collection
// index = -1 indicates a global seeding region
    int index;


  
  };
  
}  // namespace ticl

#endif
