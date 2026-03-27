#include <alpaka/alpaka.hpp>
#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace alpaka_portable_test = ALPAKA_ACCELERATOR_NAMESPACE::portabletest;
namespace alpaka_ngt = ALPAKA_ACCELERATOR_NAMESPACE::ngt;

/*
 * This test demonstrates a way to use the TrivialSerialisation mechanism to copy
 * portable types (PortableCollections, PortableObjects, and multi-block
 * collections) of which only the runtime type information is known.
 *
 * For each type, the test performs the following steps:
 * - Create a portable type and initialize it with some data, wrapped in
 *   DeviceProductType.
 * - Wrap it in edm::Wrapper and cast to WrapperBase* to hide the concrete type.
 * - Use the SerialiserFactory plugin to obtain a type-erased reader and writer.
 * - Copy the memory regions from the reader into the writer.
 * - Retrieve the cloned product and verify its contents match the original.
 *
 * Tested types: single-block PortableCollection, PortableObject, 2-block and
 * 3-block PortableCollections.
 */

namespace test {

  // helper to create a dummy EDMetadata object for the purpose of these tests.
  std::shared_ptr<EDMetadata> makeMetadata(Device const& device) {
    if constexpr (detail::useProductDirectly) {
      return std::make_shared<EDMetadata>(std::make_shared<Queue>(device));
    } else {
      return std::make_shared<EDMetadata>(std::make_shared<Queue>(device), std::make_shared<Event>(device));
    }
  }

  // helper to wrap a product in edm::Wrapper<detail::DeviceProductType<T>>.
  template <typename T>
  edm::Wrapper<detail::DeviceProductType<T>> wrapProduct(T&& product, std::shared_ptr<EDMetadata> metadata) {
    if constexpr (detail::useProductDirectly) {
      return edm::Wrapper<detail::DeviceProductType<T>>(edm::WrapperBase::Emplace{}, std::move(product));
    } else {
      // asynchronous backends, edm::DeviceProduct<T>'s constructor takes a
      // metadata object
      return edm::Wrapper<detail::DeviceProductType<T>>(
          edm::WrapperBase::Emplace{}, std::move(metadata), std::move(product));
    }
  }

  // helper to unwrap a product of type "T" from an
  // edm::Wrapper<detail::DeviceProductType<T>>
  template <typename WrapperType>
  auto const& extractProductFromEdmWrapper(WrapperType const& wrapper, EDMetadata& metadata) {
    if constexpr (detail::useProductDirectly) {
      // synchronous backends, WrapperType is edm::Wrapper<T>
      return wrapper.bareProduct();
    } else {
      // asynchronous backends, WrapperType is edm::Wrapper<edm::DeviceProduct<T>>
      return wrapper.bareProduct().template getSynchronized<EDMetadata>(metadata, true);
    }
  }
}  // namespace test

TEST_CASE("Test MemoryCopyTraits", "[MemoryCopyTraits]") {
  if (not edmplugin::PluginManager::isAvailable())
    edmplugin::PluginManager::configure(edmplugin::standard::config());

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");
  }

  SECTION("PortableCollection<Device, ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestSoALayout<128, false>>") {
    using PortableCollectionType = ::PortableCollection<Device, alpaka_portable_test::TestSoALayout<128, false>>;
    using DeviceProductType = detail::DeviceProductType<PortableCollectionType>;
    using PortableHostCollectionType = PortableHostCollection<alpaka_portable_test::TestSoALayout<128, false>>;
    const int size = 10;

    for (auto const& device : devices) {
      std::cout << "Running on " << alpaka::getName(device) << std::endl;
      Queue queue(device);

      // Create a PortableHostCollection, to be used as a reference for the checks
      PortableHostCollectionType refHostCollection(queue, size);

      // Initialize it with some data
      refHostCollection.view().r() = 3.14;
      for (int i = 0; i < size; i++) {
        refHostCollection.view()[i].x() = i * size + 1;
        refHostCollection.view()[i].y() = i * size + 2;
        refHostCollection.view()[i].z() = i * size + 3;
        refHostCollection.view()[i].id() = i;
        refHostCollection.view()[i].flags() = std::array<short, 4>{
            {static_cast<short>(i), static_cast<short>(i + 1), static_cast<short>(i + 2), static_cast<short>(i + 3)}};
        refHostCollection.view()[i].m = Eigen::Matrix<double, 3, 6>::Identity() * i;
      }

      // Function to validate a PortableCollection whose contents should be
      // identical to the reference Host collection above
      auto checkPortableCollection = [&queue, &refHostCollection](PortableCollectionType const& col) {
        // Since PortableCollection might be a device collection, copy first its
        // data into an auxiliary PortableHostCollection
        PortableHostCollectionType auxPortableHostCollection(queue, size);
        alpaka::memcpy(queue, auxPortableHostCollection.buffer(), col.buffer());
        alpaka::wait(queue);

        REQUIRE(auxPortableHostCollection.const_view().r() == refHostCollection.const_view().r());
        for (int i = 0; i < size; i++) {
          REQUIRE(auxPortableHostCollection.const_view()[i].x() == refHostCollection.const_view()[i].x());
          REQUIRE(auxPortableHostCollection.const_view()[i].y() == refHostCollection.const_view()[i].y());
          REQUIRE(auxPortableHostCollection.const_view()[i].z() == refHostCollection.const_view()[i].z());
          REQUIRE(auxPortableHostCollection.const_view()[i].id() == refHostCollection.const_view()[i].id());
          REQUIRE(auxPortableHostCollection.const_view()[i].flags() == refHostCollection.const_view()[i].flags());
          REQUIRE(auxPortableHostCollection.const_view()[i].m() == refHostCollection.const_view()[i].m());
        }
      };

      // Create a PortableCollection
      PortableCollectionType sourceCollection(queue, size);

      // Copy the reference host collection into this newly created one
      alpaka::memcpy(queue, sourceCollection.buffer(), refHostCollection.buffer());

      // Check that the collection has been successfully initialized.
      checkPortableCollection(sourceCollection);

      // Wrap it in edm::Wrapper
      auto metadata = test::makeMetadata(device);
      auto wrapper_original = test::wrapProduct<PortableCollectionType>(std::move(sourceCollection), metadata);

      // Now cast the wrapper to edm::WrapperBase, hiding the underlying collection type
      edm::WrapperBase const* wb_original = static_cast<const edm::WrapperBase*>(&wrapper_original);

      // Check that a specialization of MemoryCopyTraits exists for this type
      static_assert(::ngt::HasMemoryCopyTraits<PortableCollectionType>);

      // Get the Serialiser plugin for this type
      std::string typeName = typeid(edm::DeviceProduct<PortableCollectionType>).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiserSource{
          alpaka_ngt::SerialiserFactoryDevice::get()->create(typeName)};

      // Create a "reader" and a "writer", then clone via memory regions
      auto reader = serialiserSource->reader(*wb_original, *metadata, true);
      auto writer = serialiserSource->writer();

      writer->initialize(queue, reader->parameters());

      // Get memory regions
      auto targets = writer->regions();
      auto sources = reader->regions();

      // Check that reader and writer have the same number of memory regions. In
      // the case of a PortableCollection this should be equal to one.
      REQUIRE(sources.size() == targets.size());

      // copy each region from the source to the clone
      for (size_t i = 0; i < sources.size(); ++i) {
        REQUIRE(sources[i].data() != nullptr);
        REQUIRE(targets[i].data() != nullptr);
        REQUIRE(targets[i].size_bytes() == sources[i].size_bytes());

        auto src_view = alpaka::createView(device, sources[i].data(), sources[i].size_bytes());
        auto trg_view = alpaka::createView(device, targets[i].data(), targets[i].size_bytes());

        alpaka::memcpy(queue, trg_view, src_view);
      }

      // Retrieve the cloned product back from its type-erased form, and verify its contents
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get(metadata);
      edm::Wrapper<DeviceProductType>* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      checkPortableCollection(test::extractProductFromEdmWrapper(*cloneptr, *metadata));
    }
  }

  SECTION("DeviceProduct<PortableDeviceObject>") {
    using DeviceObjectType = alpaka_portable_test::TestDeviceObject;
    using DeviceProductType = detail::DeviceProductType<DeviceObjectType>;
    const alpaka_portable_test::TestStruct testData{5.0, 12.0, 13.0, 42};

    for (auto const& device : devices) {
      std::cout << "Running DeviceObject test on " << alpaka::getName(device) << std::endl;
      Queue queue(device);

      // Create a reference host object with known data
      PortableHostObject<alpaka_portable_test::TestStruct> hostSource(queue);
      hostSource->x = testData.x;
      hostSource->y = testData.y;
      hostSource->z = testData.z;
      hostSource->id = testData.id;

      // Create a DeviceObject, copy the reference host object into it,
      DeviceObjectType sourceObject(queue);
      alpaka::memcpy(queue, sourceObject.buffer(), hostSource.buffer());

      // Wrap it in the WrapperBase (type-erased)
      auto metadata = test::makeMetadata(device);
      auto wrapper = test::wrapProduct<DeviceObjectType>(std::move(sourceObject), metadata);
      edm::WrapperBase const* wb = static_cast<const edm::WrapperBase*>(&wrapper);

      // Get the serialiser plugin
      std::string typeName = typeid(edm::DeviceProduct<DeviceObjectType>).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiser{
          alpaka_ngt::SerialiserFactoryDevice::get()->create(typeName)};
      REQUIRE(serialiser);

      // Read and write
      auto reader = serialiser->reader(*wb, *metadata, true);
      auto writer = serialiser->writer();

      writer->initialize(queue, reader->parameters());

      auto targets = writer->regions();
      auto sources = reader->regions();
      REQUIRE(sources.size() == targets.size());

      for (size_t i = 0; i < sources.size(); ++i) {
        REQUIRE(sources[i].size_bytes() == targets[i].size_bytes());
        auto src_view = alpaka::createView(device, sources[i].data(), sources[i].size_bytes());
        auto trg_view = alpaka::createView(device, targets[i].data(), targets[i].size_bytes());
        alpaka::memcpy(queue, trg_view, src_view);
      }

      // Get the collection back from its wrapped form
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get(metadata);
      auto* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      REQUIRE(cloneptr);

      // Copy back to host to check
      PortableHostObject<alpaka_portable_test::TestStruct> hostClone(queue);
      alpaka::memcpy(queue, hostClone.buffer(), test::extractProductFromEdmWrapper(*cloneptr, *metadata).buffer());
      alpaka::wait(queue);

      REQUIRE(hostClone->x == testData.x);
      REQUIRE(hostClone->y == testData.y);
      REQUIRE(hostClone->z == testData.z);
      REQUIRE(hostClone->id == testData.id);
    }
  }

  SECTION("DeviceProduct<PortableDeviceCollection2>") {
    using DeviceCollection2Type = alpaka_portable_test::TestDeviceCollection2;
    using DeviceProductType = detail::DeviceProductType<DeviceCollection2Type>;
    using HostCollection2Type = PortableHostCollection<alpaka_portable_test::TestSoABlocks2>;
    const int size1 = 7;
    const int size2 = 11;

    for (auto const& device : devices) {
      std::cout << "Running DeviceCollection2 test on " << alpaka::getName(device) << std::endl;
      Queue queue(device);

      // Create reference data on host
      HostCollection2Type refHost(queue, size1, size2);
      refHost.view().first().r() = 1.11;
      refHost.view().second().r2() = 2.22;
      for (int i = 0; i < size1; i++) {
        refHost.view().first()[i].x() = i * 10.0;
        refHost.view().first()[i].y() = 0.0;
        refHost.view().first()[i].z() = 0.0;
        refHost.view().first()[i].id() = i;
        refHost.view().first()[i].flags() = std::array<short, 4>{{0, 0, 0, 0}};
        refHost.view().first()[i].m = Eigen::Matrix<double, 3, 6>::Zero();
      }
      for (int i = 0; i < size2; i++) {
        refHost.view().second()[i].x2() = i * 20.0;
        refHost.view().second()[i].y2() = 0.0;
        refHost.view().second()[i].z2() = 0.0;
        refHost.view().second()[i].id2() = i + 100;
        refHost.view().second()[i].m2 = Eigen::Matrix<double, 3, 6>::Zero();
      }

      // Create DeviceCollection2, fill it, and wrap in DeviceProduct
      DeviceCollection2Type sourceCollection(queue, size1, size2);
      alpaka::memcpy(queue, sourceCollection.buffer(), refHost.buffer());

      // Wrap it and cast to WrapperBase
      auto metadata = test::makeMetadata(device);
      auto wrapper = test::wrapProduct<DeviceCollection2Type>(std::move(sourceCollection), metadata);
      edm::WrapperBase const* wb = static_cast<const edm::WrapperBase*>(&wrapper);

      // Get the serialiser plugin
      std::string typeName = typeid(edm::DeviceProduct<DeviceCollection2Type>).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiser{
          alpaka_ngt::SerialiserFactoryDevice::get()->create(typeName)};
      REQUIRE(serialiser);

      // Read and write
      auto reader = serialiser->reader(*wb, *metadata, true);
      auto writer = serialiser->writer();

      writer->initialize(queue, reader->parameters());

      auto targets = writer->regions();
      auto sources_r = reader->regions();
      REQUIRE(sources_r.size() == targets.size());

      for (size_t i = 0; i < sources_r.size(); ++i) {
        REQUIRE(sources_r[i].size_bytes() == targets[i].size_bytes());
        auto src_view = alpaka::createView(device, sources_r[i].data(), sources_r[i].size_bytes());
        auto trg_view = alpaka::createView(device, targets[i].data(), targets[i].size_bytes());
        alpaka::memcpy(queue, trg_view, src_view);
      }

      // Get the clone
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get(metadata);
      auto* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      REQUIRE(cloneptr);

      // Copy back to host and check
      HostCollection2Type verifyHost(queue, size1, size2);
      alpaka::memcpy(queue, verifyHost.buffer(), test::extractProductFromEdmWrapper(*cloneptr, *metadata).buffer());
      alpaka::wait(queue);

      REQUIRE(verifyHost.const_view().first().r() == 1.11);
      REQUIRE(verifyHost.const_view().second().r2() == 2.22);
      for (int i = 0; i < size1; i++) {
        REQUIRE(verifyHost.const_view().first()[i].x() == i * 10.0);
        REQUIRE(verifyHost.const_view().first()[i].id() == i);
      }
      for (int i = 0; i < size2; i++) {
        REQUIRE(verifyHost.const_view().second()[i].x2() == i * 20.0);
        REQUIRE(verifyHost.const_view().second()[i].id2() == i + 100);
      }
    }
  }
}
