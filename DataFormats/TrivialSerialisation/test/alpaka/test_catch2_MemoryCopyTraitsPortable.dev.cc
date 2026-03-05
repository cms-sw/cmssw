#include <cstring>
#include <utility>

#include <alpaka/alpaka.hpp>

#include <catch2/catch_all.hpp>

#include <Eigen/Dense>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace alpaka_portable_test = ALPAKA_ACCELERATOR_NAMESPACE::portabletest;

/*
 * Catch2 tests for MemoryCopyTraits specializations of portable types.
 *
 * This test verifies that:
 * - PortableHostCollection has valid MemoryCopyTraits with Properties and initialize()
 * - PortableHostObject has valid MemoryCopyTraits with initialize() but no Properties
 * - PortableDeviceCollection has valid MemoryCopyTraits with Properties and initialize()
 * - PortableDeviceObject has valid MemoryCopyTraits with initialize() but no Properties
 * - Multi-block collections (SoABlocks2, SoABlocks3) have valid MemoryCopyTraits
 *
 * It also tests that data can be correctly copied using the MemoryCopyTraits interface
 * for both host and device portable types.
 */

TEST_CASE("Test MemoryCopyTraits with portable types", "[MemoryCopyTraits][Portable]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");
  }

  SECTION("PortableHostCollection checks") {
    using HostCollectionType = PortableHostCollection<alpaka_portable_test::TestSoALayout<128, false>>;

    // PortableHostCollection should have MemoryCopyTraits
    static_assert(::ngt::HasMemoryCopyTraits<HostCollectionType>);
    // It should have Properties (the size)
    static_assert(::ngt::HasTrivialCopyProperties<HostCollectionType>);
    // It should have a valid initialize()
    static_assert(::ngt::HasValidInitialize<HostCollectionType>);
    // It should have regions
    static_assert(::ngt::HasRegions<HostCollectionType>);
  }

  SECTION("PortableHostObject checks") {
    using HostObjectType = PortableHostObject<alpaka_portable_test::TestStruct>;

    // PortableHostObject should have MemoryCopyTraits
    static_assert(::ngt::HasMemoryCopyTraits<HostObjectType>);
    // It should not have Properties
    static_assert(not ::ngt::HasTrivialCopyProperties<HostObjectType>);
    // It should have regions
    static_assert(::ngt::HasRegions<HostObjectType>);
  }

  SECTION("PortableDeviceCollection checks") {
    using DeviceCollectionType = alpaka_portable_test::TestDeviceCollection;

    // PortableDeviceCollection should have MemoryCopyTraits
    static_assert(::ngt::HasMemoryCopyTraits<DeviceCollectionType>);
    // It should have Properties (the size)
    static_assert(::ngt::HasTrivialCopyProperties<DeviceCollectionType>);
    // It should have regions
    static_assert(::ngt::HasRegions<DeviceCollectionType>);
  }

  SECTION("PortableDeviceObject checks") {
    using DeviceObjectType = alpaka_portable_test::TestDeviceObject;

    // PortableDeviceObject should have MemoryCopyTraits
    static_assert(::ngt::HasMemoryCopyTraits<DeviceObjectType>);
    // It should not have Properties
    static_assert(not ::ngt::HasTrivialCopyProperties<DeviceObjectType>);
    // It should have regions
    static_assert(::ngt::HasRegions<DeviceObjectType>);
  }

  SECTION("Multi-block PortableHostCollection checks") {
    // SoABlocks2
    using HostCollection2Type = PortableHostCollection<alpaka_portable_test::TestSoABlocks2>;
    static_assert(::ngt::HasMemoryCopyTraits<HostCollection2Type>);
    static_assert(::ngt::HasTrivialCopyProperties<HostCollection2Type>);
    static_assert(::ngt::HasRegions<HostCollection2Type>);

    // SoABlocks3
    using HostCollection3Type = PortableHostCollection<alpaka_portable_test::TestSoABlocks3>;
    static_assert(::ngt::HasMemoryCopyTraits<HostCollection3Type>);
    static_assert(::ngt::HasTrivialCopyProperties<HostCollection3Type>);
    static_assert(::ngt::HasRegions<HostCollection3Type>);
  }

  SECTION("Multi-block PortableDeviceCollection checks") {
    using DeviceCollection2Type = alpaka_portable_test::TestDeviceCollection2;
    static_assert(::ngt::HasMemoryCopyTraits<DeviceCollection2Type>);
    static_assert(::ngt::HasTrivialCopyProperties<DeviceCollection2Type>);
    static_assert(::ngt::HasRegions<DeviceCollection2Type>);

    using DeviceCollection3Type = alpaka_portable_test::TestDeviceCollection3;
    static_assert(::ngt::HasMemoryCopyTraits<DeviceCollection3Type>);
    static_assert(::ngt::HasTrivialCopyProperties<DeviceCollection3Type>);
    static_assert(::ngt::HasRegions<DeviceCollection3Type>);
  }

  SECTION("PortableHostCollection copy via MemoryCopyTraits") {
    using HostCollectionType = PortableHostCollection<alpaka_portable_test::TestSoALayout<128, false>>;
    const int size = 10;

    for (auto const& device : devices) {
      Queue queue(device);

      // Create a source collection and fill it with data
      HostCollectionType source(queue, size);
      source.view().r() = 3.14;
      for (int i = 0; i < size; i++) {
        source.view()[i].x() = i * 10.0 + 1;
        source.view()[i].y() = i * 10.0 + 2;
        source.view()[i].z() = i * 10.0 + 3;
        source.view()[i].id() = i;
        source.view()[i].flags() = std::array<short, 4>{
            {static_cast<short>(i), static_cast<short>(i + 1), static_cast<short>(i + 2), static_cast<short>(i + 3)}};
        source.view()[i].m = Eigen::Matrix<double, 3, 6>::Identity() * i;
      }

      // Get properties from source
      auto props = ::ngt::MemoryCopyTraits<HostCollectionType>::properties(source);
      REQUIRE(props == size);

      // Create a clone and initialize it
      HostCollectionType clone(edm::kUninitialized);
      ::ngt::MemoryCopyTraits<HostCollectionType>::initialize(clone, props);

      // Get regions and copy
      auto sources_regions = ::ngt::MemoryCopyTraits<HostCollectionType>::regions(std::as_const(source));
      auto targets_regions = ::ngt::MemoryCopyTraits<HostCollectionType>::regions(clone);

      REQUIRE(sources_regions.size() == targets_regions.size());
      REQUIRE(sources_regions.size() == 1);
      REQUIRE(sources_regions[0].size_bytes() == targets_regions[0].size_bytes());

      std::memcpy(targets_regions[0].data(), sources_regions[0].data(), sources_regions[0].size_bytes());

      // Verify clone
      REQUIRE(clone.const_view().r() == source.const_view().r());
      for (int i = 0; i < size; i++) {
        REQUIRE(clone.const_view()[i].x() == source.const_view()[i].x());
        REQUIRE(clone.const_view()[i].y() == source.const_view()[i].y());
        REQUIRE(clone.const_view()[i].z() == source.const_view()[i].z());
        REQUIRE(clone.const_view()[i].id() == source.const_view()[i].id());
        REQUIRE(clone.const_view()[i].flags() == source.const_view()[i].flags());
        REQUIRE(clone.const_view()[i].m() == source.const_view()[i].m());
      }
    }
  }

  SECTION("PortableHostObject copy via MemoryCopyTraits") {
    using HostObjectType = PortableHostObject<alpaka_portable_test::TestStruct>;

    for (auto const& device : devices) {
      Queue queue(device);

      // Create source object
      HostObjectType source(cms::alpakatools::host());
      source->x = 5.0;
      source->y = 12.0;
      source->z = 13.0;
      source->id = 42;

      // Create an uninitialized clone
      HostObjectType clone(edm::kUninitialized);
      ::ngt::MemoryCopyTraits<HostObjectType>::initialize(clone);

      // Get regions and copy
      auto sources_regions = ::ngt::MemoryCopyTraits<HostObjectType>::regions(std::as_const(source));
      auto targets_regions = ::ngt::MemoryCopyTraits<HostObjectType>::regions(clone);

      REQUIRE(sources_regions.size() == targets_regions.size());
      REQUIRE(sources_regions.size() == 1);
      REQUIRE(sources_regions[0].size_bytes() == sizeof(alpaka_portable_test::TestStruct));
      REQUIRE(targets_regions[0].size_bytes() == sizeof(alpaka_portable_test::TestStruct));

      std::memcpy(targets_regions[0].data(), sources_regions[0].data(), sources_regions[0].size_bytes());

      // Verify clone
      REQUIRE(clone->x == 5.0);
      REQUIRE(clone->y == 12.0);
      REQUIRE(clone->z == 13.0);
      REQUIRE(clone->id == 42);
    }
  }

  SECTION("PortableDeviceCollection copy via MemoryCopyTraits") {
    using DeviceCollectionType = alpaka_portable_test::TestDeviceCollection;
    using HostCollectionType = PortableHostCollection<alpaka_portable_test::TestSoALayout<128, false>>;
    const int size = 10;

    for (auto const& device : devices) {
      Queue queue(device);

      // Create a reference host collection with known data
      HostCollectionType refHost(queue, size);
      refHost.view().r() = 2.71;
      for (int i = 0; i < size; i++) {
        refHost.view()[i].x() = i * 100.0;
        refHost.view()[i].y() = i * 200.0;
        refHost.view()[i].z() = i * 300.0;
        refHost.view()[i].id() = i + 100;
        refHost.view()[i].flags() = std::array<short, 4>{
            {static_cast<short>(i), static_cast<short>(i + 1), static_cast<short>(i + 2), static_cast<short>(i + 3)}};
        refHost.view()[i].m = Eigen::Matrix<double, 3, 6>::Identity() * i;
      }

      // Create a device collection and copy the reference data to it
      DeviceCollectionType source(queue, size);
      alpaka::memcpy(queue, source.buffer(), refHost.buffer());
      alpaka::wait(queue);

      // Get properties
      auto props = ::ngt::MemoryCopyTraits<DeviceCollectionType>::properties(source);
      REQUIRE(props == size);

      // Create a clone on device
      DeviceCollectionType clone(edm::kUninitialized);
      ::ngt::MemoryCopyTraits<DeviceCollectionType>::initialize(queue, clone, props);

      // Get regions and copy (device-to-device via alpaka)
      auto sources_regions = ::ngt::MemoryCopyTraits<DeviceCollectionType>::regions(std::as_const(source));
      auto targets_regions = ::ngt::MemoryCopyTraits<DeviceCollectionType>::regions(clone);

      REQUIRE(sources_regions.size() == targets_regions.size());
      for (size_t i = 0; i < sources_regions.size(); ++i) {
        REQUIRE(sources_regions[i].size_bytes() == targets_regions[i].size_bytes());
        auto src_view = alpaka::createView(device, sources_regions[i].data(), sources_regions[i].size_bytes());
        auto trg_view = alpaka::createView(device, targets_regions[i].data(), targets_regions[i].size_bytes());
        alpaka::memcpy(queue, trg_view, src_view);
      }
      alpaka::wait(queue);

      // Copy the clone back to host to verify
      HostCollectionType verifyHost(queue, size);
      alpaka::memcpy(queue, verifyHost.buffer(), clone.buffer());
      alpaka::wait(queue);

      REQUIRE(verifyHost.const_view().r() == refHost.const_view().r());
      for (int i = 0; i < size; i++) {
        REQUIRE(verifyHost.const_view()[i].x() == refHost.const_view()[i].x());
        REQUIRE(verifyHost.const_view()[i].y() == refHost.const_view()[i].y());
        REQUIRE(verifyHost.const_view()[i].z() == refHost.const_view()[i].z());
        REQUIRE(verifyHost.const_view()[i].id() == refHost.const_view()[i].id());
        REQUIRE(verifyHost.const_view()[i].flags() == refHost.const_view()[i].flags());
      }
    }
  }

  SECTION("PortableDeviceObject copy via MemoryCopyTraits") {
    using DeviceObjectType = alpaka_portable_test::TestDeviceObject;

    for (auto const& device : devices) {
      Queue queue(device);

      // Create source object on host, copy to device
      PortableHostObject<alpaka_portable_test::TestStruct> hostSource(queue);
      hostSource->x = 1.0;
      hostSource->y = 2.0;
      hostSource->z = 3.0;
      hostSource->id = 99;

      DeviceObjectType source(queue);
      alpaka::memcpy(queue, source.buffer(), hostSource.buffer());
      alpaka::wait(queue);

      // Create an uninitialized clone
      DeviceObjectType clone(edm::kUninitialized);
      ::ngt::MemoryCopyTraits<DeviceObjectType>::initialize(queue, clone);

      // Get regions and copy
      auto sources_regions = ::ngt::MemoryCopyTraits<DeviceObjectType>::regions(std::as_const(source));
      auto targets_regions = ::ngt::MemoryCopyTraits<DeviceObjectType>::regions(clone);

      REQUIRE(sources_regions.size() == 1);
      REQUIRE(targets_regions.size() == 1);
      REQUIRE(sources_regions[0].size_bytes() == sizeof(alpaka_portable_test::TestStruct));

      auto src_view = alpaka::createView(device, sources_regions[0].data(), sources_regions[0].size_bytes());
      auto trg_view = alpaka::createView(device, targets_regions[0].data(), targets_regions[0].size_bytes());
      alpaka::memcpy(queue, trg_view, src_view);
      alpaka::wait(queue);

      // Copy clone back to host and verify
      PortableHostObject<alpaka_portable_test::TestStruct> hostClone(queue);
      alpaka::memcpy(queue, hostClone.buffer(), clone.buffer());
      alpaka::wait(queue);

      REQUIRE(hostClone->x == 1.0);
      REQUIRE(hostClone->y == 2.0);
      REQUIRE(hostClone->z == 3.0);
      REQUIRE(hostClone->id == 99);
    }
  }

  SECTION("Multi-block PortableHostCollection copy via MemoryCopyTraits") {
    using HostCollection2Type = PortableHostCollection<alpaka_portable_test::TestSoABlocks2>;
    const int size1 = 8;
    const int size2 = 12;

    for (auto const& device : devices) {
      Queue queue(device);

      // Create source with two blocks
      HostCollection2Type source(queue, size1, size2);
      source.view().first().r() = 1.23;
      source.view().second().r2() = 4.56;
      for (int i = 0; i < size1; i++) {
        source.view().first()[i].x() = i * 1.0;
        source.view().first()[i].id() = i;
      }
      for (int i = 0; i < size2; i++) {
        source.view().second()[i].x2() = i * 2.0;
        source.view().second()[i].id2() = i + 100;
      }

      // Get properties
      auto props = ::ngt::MemoryCopyTraits<HostCollection2Type>::properties(source);
      REQUIRE(props[0] == size1);
      REQUIRE(props[1] == size2);

      // Create clone and initialize
      HostCollection2Type clone(edm::kUninitialized);
      ::ngt::MemoryCopyTraits<HostCollection2Type>::initialize(clone, props);

      // Copy regions
      auto sources_regions = ::ngt::MemoryCopyTraits<HostCollection2Type>::regions(std::as_const(source));
      auto targets_regions = ::ngt::MemoryCopyTraits<HostCollection2Type>::regions(clone);

      REQUIRE(sources_regions.size() == targets_regions.size());
      for (size_t i = 0; i < sources_regions.size(); ++i) {
        REQUIRE(sources_regions[i].size_bytes() == targets_regions[i].size_bytes());
        std::memcpy(targets_regions[i].data(), sources_regions[i].data(), sources_regions[i].size_bytes());
      }

      // Verify
      REQUIRE(clone.const_view().first().r() == 1.23);
      REQUIRE(clone.const_view().second().r2() == 4.56);
      for (int i = 0; i < size1; i++) {
        REQUIRE(clone.const_view().first()[i].x() == i * 1.0);
        REQUIRE(clone.const_view().first()[i].id() == i);
      }
      for (int i = 0; i < size2; i++) {
        REQUIRE(clone.const_view().second()[i].x2() == i * 2.0);
        REQUIRE(clone.const_view().second()[i].id2() == i + 100);
      }
    }
  }
}
