#include "catch.hpp"

#include "DataFormats/Provenance/interface/ResourcesDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

TEST_CASE("ResourcesDescription", "[ResourcesDescription]") {
  SECTION("Construction from empty string") { CHECK(edm::ResourcesDescription("") == edm::ResourcesDescription()); }

  SECTION("Default construction") {
    edm::ResourcesDescription resources;
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("Microarchitecture") {
    edm::ResourcesDescription resources;
    resources.microarchitecture = "x86-64-v3";
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("CPU models") {
    edm::ResourcesDescription resources;
    resources.cpuModels = {"Intel something", "AMD something else"};
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("accelerators") {
    edm::ResourcesDescription resources;
    resources.selectedAccelerators = {"cpu", "gpu"};
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("GPU models") {
    edm::ResourcesDescription resources;
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("All fields") {
    edm::ResourcesDescription resources;
    resources.microarchitecture = "x86-64-v3";
    resources.cpuModels = {"Intel something", "AMD something else"};
    resources.selectedAccelerators = {"cpu", "gpu"};
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    CHECK(edm::ResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("Serialization has additional things (forward compatibility)") {
    edm::ResourcesDescription resources, resources2;
    resources.microarchitecture = "x86-64-v3";
    resources.cpuModels = {"Intel something", "AMD something else"};
    resources.selectedAccelerators = {"cpu", "gpu"};
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    resources2.microarchitecture = "this";
    resources2.cpuModels = {"is"};
    resources2.selectedAccelerators = {"something"};
    resources2.gpuModels = {"else"};
    auto const serial = resources.serialize() + resources2.serialize();
    CHECK(edm::ResourcesDescription(serial) == resources);
  }

  SECTION("Error cases") {
    SECTION("Invalid serialized string") {
      CHECK_THROWS_AS(edm::ResourcesDescription("foo"), edm::Exception);

      edm::ResourcesDescription resources;
      resources.microarchitecture = "x86-64-v3";
      auto serialized = resources.serialize();
      serialized.back() = ',';
      CHECK_THROWS_AS(edm::ResourcesDescription(serialized), edm::Exception);
      serialized.pop_back();
      CHECK_THROWS_AS(edm::ResourcesDescription(serialized), edm::Exception);
    }
  }
}
