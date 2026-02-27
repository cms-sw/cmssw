#include "catch.hpp"

#include "DataFormats/Provenance/interface/HardwareResourcesDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

TEST_CASE("HardwareResourcesDescription", "[HardwareResourcesDescription]") {
  SECTION("Construction from empty string") {
    CHECK(edm::HardwareResourcesDescription("") == edm::HardwareResourcesDescription());
  }

  SECTION("Default construction") {
    edm::HardwareResourcesDescription resources;
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
    CHECK(resources.serialize().empty());
  }

  SECTION("Microarchitecture") {
    edm::HardwareResourcesDescription resources;
    resources.microarchitecture = "x86-64-v3";
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("CPU models") {
    edm::HardwareResourcesDescription resources;
    resources.cpuModels = {"Intel something", "AMD something else"};
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("accelerators") {
    edm::HardwareResourcesDescription resources;
    resources.selectedAccelerators = {"cpu", "gpu"};
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("GPU models") {
    edm::HardwareResourcesDescription resources;
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("All fields") {
    edm::HardwareResourcesDescription resources;
    resources.microarchitecture = "x86-64-v3";
    resources.cpuModels = {"Intel something", "AMD something else"};
    resources.selectedAccelerators = {"cpu", "gpu"};
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    CHECK(edm::HardwareResourcesDescription(resources.serialize()) == resources);
  }

  SECTION("Serialization has additional things (forward compatibility)") {
    edm::HardwareResourcesDescription resources, resources2;
    resources.microarchitecture = "x86-64-v3";
    resources.cpuModels = {"Intel something", "AMD something else"};
    resources.selectedAccelerators = {"cpu", "gpu"};
    resources.gpuModels = {"NVIDIA something", "NVIDIA something else"};
    resources2.microarchitecture = "this";
    resources2.cpuModels = {"is"};
    resources2.selectedAccelerators = {"something"};
    resources2.gpuModels = {"else"};
    auto const serial = resources.serialize() + resources2.serialize();
    CHECK(edm::HardwareResourcesDescription(serial) == resources);
  }

  SECTION("Error cases") {
    SECTION("Invalid serialized string") {
      CHECK_THROWS_AS(edm::HardwareResourcesDescription("foo"), edm::Exception);

      edm::HardwareResourcesDescription resources;
      resources.microarchitecture = "x86-64-v3";
      auto serialized = resources.serialize();
      SECTION("Last container does not have the delimiter") {
        serialized.back() = ',';
        CHECK_THROWS_AS(edm::HardwareResourcesDescription(serialized), edm::Exception);
      }
      SECTION("Too few containers") {
        serialized.pop_back();
        CHECK_THROWS_AS(edm::HardwareResourcesDescription(serialized), edm::Exception);
      }
    }
  }
}
