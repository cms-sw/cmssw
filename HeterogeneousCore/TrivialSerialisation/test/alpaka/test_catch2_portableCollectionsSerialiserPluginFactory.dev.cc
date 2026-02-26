#include <alpaka/alpaka.hpp>
#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

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

namespace {
  template <typename WrapperType>
  auto const& extractProduct(WrapperType const& w) {
    if constexpr (detail::useProductDirectly) {
      return w.bareProduct();
    } else {
      return w.bareProduct().template getSynchronized<EDMetadata>();
    }
  }
}  // namespace

TEST_CASE("Test MemoryCopyTraits", "[MemoryCopyTraits]") {
  // initialize the edmplugin::PluginManager
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

      // Create a PortableHostCollection, to be used as a reference for all checks
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

        printf("Comparing to the reference host collection\n");
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

      // Create a PortableCollection, fill it, and wrap it in DeviceProduct
      PortableCollectionType sourceCollection(queue, size);

      // Copy the reference host collection into this newly created one
      alpaka::memcpy(queue, sourceCollection.buffer(), refHostCollection.buffer());
      alpaka::wait(queue);

      // Check that the collection has been successfully initialized.
      checkPortableCollection(sourceCollection);

      // Wrap in edm::Wrapper<DeviceProduct<PortableCollectionType>>
      edm::Wrapper<DeviceProductType> wrapper_original(
          std::make_unique<DeviceProductType>(std::move(sourceCollection)));

      // Now cast the wrapper to edm::WrapperBase, hiding the underlying collection type
      edm::WrapperBase const* wb_original = static_cast<const edm::WrapperBase*>(&wrapper_original);

      // Check that a Serialiser plugin exists for this type
      static_assert(::ngt::HasMemoryCopyTraits<PortableCollectionType>);

      // Get the plugin (registered under the type name)
      std::string typeName = typeid(DeviceProductType).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiserSource{
          alpaka_ngt::SerialiserFactoryPortable::get()->create(typeName)};

      // Create Reader and Writer, then clone via memory regions
      auto reader = serialiserSource->reader(*wb_original);
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
      alpaka::wait(queue);

      // Retrieve the cloned product back from its type-erased form, and verify its contents
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get();
      edm::Wrapper<DeviceProductType>* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      checkPortableCollection(extractProduct(*cloneptr));
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
      // and wrap in DeviceProduct
      DeviceObjectType sourceObject(queue);
      alpaka::memcpy(queue, sourceObject.buffer(), hostSource.buffer());
      alpaka::wait(queue);

      // Wrap it in the WrapperBase (type-erased)
      edm::Wrapper<DeviceProductType> wrapper(std::make_unique<DeviceProductType>(std::move(sourceObject)));
      edm::WrapperBase const* wb = static_cast<const edm::WrapperBase*>(&wrapper);

      // Get the serialiser plugin
      std::string typeName = typeid(DeviceProductType).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiser{
          alpaka_ngt::SerialiserFactoryPortable::get()->create(typeName)};
      REQUIRE(serialiser);

      // Read and write
      auto reader = serialiser->reader(*wb);
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
      alpaka::wait(queue);

      // Get the collection back from its wrapped form
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get();
      auto* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      REQUIRE(cloneptr);

      // Copy back to host to check
      PortableHostObject<alpaka_portable_test::TestStruct> hostClone(queue);
      alpaka::memcpy(queue, hostClone.buffer(), extractProduct(*cloneptr).buffer());
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
      alpaka::wait(queue);

      // Wrap it and cast to WrapperBase
      edm::Wrapper<DeviceProductType> wrapper(std::make_unique<DeviceProductType>(std::move(sourceCollection)));
      edm::WrapperBase const* wb = static_cast<const edm::WrapperBase*>(&wrapper);

      // Get the serialiser plugin
      std::string typeName = typeid(DeviceProductType).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiser{
          alpaka_ngt::SerialiserFactoryPortable::get()->create(typeName)};
      REQUIRE(serialiser);

      // Read and write
      auto reader = serialiser->reader(*wb);
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
      alpaka::wait(queue);

      // Get clone and verify
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get();
      auto* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      REQUIRE(cloneptr);

      // Copy back to host and check
      HostCollection2Type verifyHost(queue, size1, size2);
      alpaka::memcpy(queue, verifyHost.buffer(), extractProduct(*cloneptr).buffer());
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

  SECTION("DeviceProduct<PortableDeviceCollection3>") {
    using DeviceCollection3Type = alpaka_portable_test::TestDeviceCollection3;
    using DeviceProductType = detail::DeviceProductType<DeviceCollection3Type>;
    using HostCollection3Type = PortableHostCollection<alpaka_portable_test::TestSoABlocks3>;
    const int size1 = 5;
    const int size2 = 8;
    const int size3 = 3;

    for (auto const& device : devices) {
      std::cout << "Running DeviceCollection3 test on " << alpaka::getName(device) << std::endl;
      Queue queue(device);

      // Create reference data on host
      HostCollection3Type refHost(queue, size1, size2, size3);
      refHost.view().first().r() = 7.77;
      refHost.view().second().r2() = 8.88;
      refHost.view().third().r3() = 9.99;
      for (int i = 0; i < size1; i++) {
        refHost.view().first()[i].x() = i;
        refHost.view().first()[i].y() = 0.0;
        refHost.view().first()[i].z() = 0.0;
        refHost.view().first()[i].id() = i;
        refHost.view().first()[i].flags() = std::array<short, 4>{{0, 0, 0, 0}};
        refHost.view().first()[i].m = Eigen::Matrix<double, 3, 6>::Zero();
      }
      for (int i = 0; i < size2; i++) {
        refHost.view().second()[i].x2() = i * 3.0;
        refHost.view().second()[i].y2() = 0.0;
        refHost.view().second()[i].z2() = 0.0;
        refHost.view().second()[i].id2() = i;
        refHost.view().second()[i].m2 = Eigen::Matrix<double, 3, 6>::Zero();
      }
      for (int i = 0; i < size3; i++) {
        refHost.view().third()[i].x3() = i * 5.0;
        refHost.view().third()[i].y3() = 0.0;
        refHost.view().third()[i].z3() = 0.0;
        refHost.view().third()[i].id3() = i;
        refHost.view().third()[i].m3 = Eigen::Matrix<double, 3, 6>::Zero();
      }

      // Create DeviceCollection3, fill it, and wrap in DeviceProduct
      DeviceCollection3Type sourceCollection(queue, size1, size2, size3);
      alpaka::memcpy(queue, sourceCollection.buffer(), refHost.buffer());
      alpaka::wait(queue);

      // Wrap it
      edm::Wrapper<DeviceProductType> wrapper(std::make_unique<DeviceProductType>(std::move(sourceCollection)));
      edm::WrapperBase const* wb = static_cast<const edm::WrapperBase*>(&wrapper);

      // Get the serialiser plugin
      std::string typeName = typeid(DeviceProductType).name();
      std::unique_ptr<alpaka_ngt::SerialiserBase> serialiser{
          alpaka_ngt::SerialiserFactoryPortable::get()->create(typeName)};
      REQUIRE(serialiser);

      auto reader = serialiser->reader(*wb);
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
      alpaka::wait(queue);

      // Get clone and verify
      std::unique_ptr<edm::WrapperBase> clonebase = writer->get();
      auto* cloneptr = dynamic_cast<edm::Wrapper<DeviceProductType>*>(clonebase.get());
      REQUIRE(cloneptr);

      HostCollection3Type verifyHost(queue, size1, size2, size3);
      alpaka::memcpy(queue, verifyHost.buffer(), extractProduct(*cloneptr).buffer());
      alpaka::wait(queue);

      REQUIRE(verifyHost.const_view().first().r() == 7.77);
      REQUIRE(verifyHost.const_view().second().r2() == 8.88);
      REQUIRE(verifyHost.const_view().third().r3() == 9.99);
      for (int i = 0; i < size1; i++) {
        REQUIRE(verifyHost.const_view().first()[i].x() == static_cast<double>(i));
        REQUIRE(verifyHost.const_view().first()[i].id() == i);
      }
      for (int i = 0; i < size2; i++) {
        REQUIRE(verifyHost.const_view().second()[i].x2() == i * 3.0);
        REQUIRE(verifyHost.const_view().second()[i].id2() == i);
      }
      for (int i = 0; i < size3; i++) {
        REQUIRE(verifyHost.const_view().third()[i].x3() == i * 5.0);
        REQUIRE(verifyHost.const_view().third()[i].id3() == i);
      }
    }
  }
}
