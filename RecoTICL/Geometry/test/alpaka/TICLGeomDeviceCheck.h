#ifndef RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h
#define RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h

#include <cstdint>

#include "CondFormats/HGCalObjects/interface/TICLGeomLookupSoA.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ticlgeomtest {

    // Runs a kernel that looks up every cell by its rawDetId, both by binary
    // search (ticlgeom::indexOf) and through the dense id hash table
    // (ticlgeom::denseIdOf); returns the number of cells whose lookups do
    // not give back their own index.
    int32_t checkLookup(Queue& queue, TICLGeomCommonSoAConstView common, TICLGeomLookupSoAConstView lookup);

  }  // namespace ticlgeomtest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoHGCal_TICL_test_alpaka_TICLGeomDeviceCheck_h
