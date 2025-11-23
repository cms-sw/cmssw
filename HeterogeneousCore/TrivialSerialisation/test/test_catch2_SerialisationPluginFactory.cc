#include <cstring>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

/*
* This test demonstrates how TrivialSerialiser can be utilized to copy supported types.
*/

TEST_CASE("Test MemoryCopyTraits", "[MemoryCopyTraits]") {
  // initialize the edmplugin::PluginManager
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  SECTION("std::vector<float>") {
    const int size = 10;

    // Create a vector.
    std::vector<float> vec(size);

    // Initialize the vector with some data.
    for (int i = 0; i < size; i++) {
      vec[i] = static_cast<float>(i);
    }

    // Wrap the vector.
    edm::Wrapper<std::vector<float>> wrapper(std::make_unique<std::vector<float>>(std::move(vec)));

    // Now cast the wrapper to edm::WrapperBase, hiding the underlying collection type.
    edm::WrapperBase const& wrapperbase = wrapper;

    // Check that the MemoryCopyTraits are specialised for the type "std::vector<float>".
    static_assert(ngt::HasMemoryCopyTraits<std::vector<float>>);

    // Get the Serialiser plugin for the type "std::vector<float>".
    std::string typeName = typeid(std::vector<float>).name();
    std::unique_ptr<ngt::SerialiserBase> serialiser = ngt::SerialiserFactory::get()->tryToCreate(typeName);
    if (serialiser) {
      INFO("A serialiser plugin has been found for type " << typeName << ".");
    } else {
      INFO("No serialiser plugin found for type " << typeName << ".");
    }
    REQUIRE(serialiser);

    // Initialize the reader and writer.
    auto reader = serialiser->reader(wrapperbase);
    auto writer = serialiser->writer();

    // Initialize the clone with properties from the original.
    writer->initialize(reader->parameters());

    // Get the memory regions.
    auto targets = writer->regions();
    auto sources = reader->regions();

    // Check that reader and writer have the same number of memory regions. In the case of a PortableCollection this should be equal to one.
    REQUIRE(sources.size() == targets.size());

    // Copy each region from the source to the clone.
    for (size_t i = 0; i < sources.size(); ++i) {
      REQUIRE(sources[i].data() != nullptr);
      REQUIRE(targets[i].data() != nullptr);
      REQUIRE(targets[i].size_bytes() == sources[i].size_bytes());
      std::memcpy(targets[i].data(), sources[i].data(), sources[i].size_bytes());
    }

    // Check that the copy succeeded.
    std::unique_ptr<edm::WrapperBase> clonebase = writer->get();
    edm::Wrapper<std::vector<float>>* cloneptr = dynamic_cast<edm::Wrapper<std::vector<float>>*>(clonebase.get());
    const std::vector<float>& clone = cloneptr->bareProduct();
    for (int i = 0; i < size; i++) {
      REQUIRE(clone[i] == static_cast<float>(i));
    }
  }
}
