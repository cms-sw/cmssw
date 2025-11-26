#ifndef DataFormats_PortableTestObjects_interface_alpaka_ParticleDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_ParticleDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace portabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::portabletest;

    using ParticleDeviceCollection = PortableCollection<ParticleSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::ParticleDeviceCollection, portabletest::ParticleHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_ParticleDeviceCollection_h
