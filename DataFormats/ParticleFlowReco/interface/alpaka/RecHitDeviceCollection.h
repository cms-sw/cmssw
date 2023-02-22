#ifndef ParticleFlowReco_RecHitDeviceCollection_h
#define ParticleFlowReco_RecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/RecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portableRecHitSoA {

    // make the names from the top-level portableRecHitSoA namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portableRecHitSoA namespace
    using namespace ::portableRecHitSoA;

    using RecHitDeviceCollection = PortableCollection<RecHitSoA>;

  }  // namespace portableRecHitSoA

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif