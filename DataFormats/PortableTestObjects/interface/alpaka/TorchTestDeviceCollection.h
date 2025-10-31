#ifndef DataFormats_PortableTestObjects_interface_alpaka_TorchTestDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_TorchTestDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TorchTestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TorchTestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace torchportabletest {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::torchportabletest;

    using ParticleDeviceCollection = PortableCollection<ParticleSoA>;
    using SimpleNetDeviceCollection = PortableCollection<SimpleNetSoA>;
    using MultiHeadNetDeviceCollection = PortableCollection<MultiHeadNetSoA>;
    using ImageDeviceCollection = PortableCollection<Image>;
    using LogitsDeviceCollection = PortableCollection<Logits>;

  }  // namespace torchportabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// heterogeneous ml data checks
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ParticleDeviceCollection,
                                      torchportabletest::ParticleHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::SimpleNetDeviceCollection,
                                      torchportabletest::SimpleNetHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::MultiHeadNetDeviceCollection,
                                      torchportabletest::MultiHeadNetHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ImageDeviceCollection, torchportabletest::ImageHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::LogitsDeviceCollection,
                                      torchportabletest::LogitsHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TorchTestDeviceCollection_h
