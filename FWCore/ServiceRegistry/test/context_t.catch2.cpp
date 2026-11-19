#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include "catch2/catch_all.hpp"

TEST_CASE("Context type matches stored context", "[Context]") {
  SECTION("ParentContext") {
    edm::ParentContext context;
    REQUIRE(context.type() == edm::ParentContext::Type::kInvalid);

    edm::ParentContext global(static_cast<edm::GlobalContext const*>(nullptr));
    REQUIRE(global.type() == edm::ParentContext::Type::kGlobal);

    edm::ParentContext internal(static_cast<edm::InternalContext const*>(nullptr));
    REQUIRE(internal.type() == edm::ParentContext::Type::kInternal);

    edm::ParentContext module(static_cast<edm::ModuleCallingContext const*>(nullptr));
    REQUIRE(module.type() == edm::ParentContext::Type::kModule);

    edm::ParentContext placeInPath(static_cast<edm::PlaceInPathContext const*>(nullptr));
    REQUIRE(placeInPath.type() == edm::ParentContext::Type::kPlaceInPath);

    edm::ParentContext stream(static_cast<edm::StreamContext const*>(nullptr));
    REQUIRE(stream.type() == edm::ParentContext::Type::kStream);
  }

  SECTION("ESParentContext") {
    edm::ESParentContext context;
    REQUIRE(context.type() == edm::ESParentContext::Type::kInvalid);

    edm::ESParentContext module(static_cast<edm::ModuleCallingContext const*>(nullptr));
    REQUIRE(module.type() == edm::ESParentContext::Type::kModule);

    edm::ESParentContext esmodule(static_cast<edm::ESModuleCallingContext const*>(nullptr));
    REQUIRE(esmodule.type() == edm::ESParentContext::Type::kESModule);
  }
}
