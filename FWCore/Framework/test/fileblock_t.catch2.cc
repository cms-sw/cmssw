#include "catch2/catch_all.hpp"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include <array>
#include <memory>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<edm::FileBlock>);
static_assert(!std::is_copy_assignable_v<edm::FileBlock>);
static_assert(!std::is_move_constructible_v<edm::FileBlock>);
static_assert(!std::is_move_assignable_v<edm::FileBlock>);
static_assert(std::has_virtual_destructor_v<edm::FileBlock>);

namespace {
  std::array<bool, edm::NumBranchTypes> const dropped{{true, false, true, false}};

  edm::FileBlock makeBlock(int reasons, std::shared_ptr<edm::ProductDependencies> const& dependencies) {
    return edm::FileBlock(edm::FileFormatVersion(17), reasons, dropped, "input.root", true, true, dependencies);
  }

  void checkCommon(edm::FileBlock const& block, int reasons, edm::ProductDependencies const* dependencies) {
    CHECK(block.fileFormatVersion().value() == 17);
    CHECK(block.whyNotFastClonable() == reasons);
    CHECK(block.hasNewlyDroppedBranch() == dropped);
    CHECK(block.fileName() == "input.root");
    CHECK(block.branchListIndexesUnchanged());
    CHECK(block.modifiedIDs());
    CHECK(&block.productDependencies() == dependencies);
  }

}  // namespace

TEST_CASE("FileBlock base defaults", "[FileBlock]") {
  edm::FileBlock block;
  CHECK(block.fileFormatVersion().value() == -1);
  CHECK(block.whyNotFastClonable() == edm::FileBlock::NoRootInputSource);
  for (auto value : block.hasNewlyDroppedBranch()) {
    CHECK_FALSE(value);
  }
  CHECK(block.fileName().empty());
  CHECK_FALSE(block.branchListIndexesUnchanged());
  CHECK_FALSE(block.modifiedIDs());
  CHECK(block.productDependencies().childLookup().empty());
}

TEST_CASE("FileBlock base construction and close", "[FileBlock]") {
  auto dependencies = std::make_shared<edm::ProductDependencies>();
  auto const reasons = edm::FileBlock::FileTooOld | edm::FileBlock::EventsToBeSorted;
  for (int initial : {0, reasons}) {
    auto block = makeBlock(initial, dependencies);
    checkCommon(block, initial, dependencies.get());
    block.close();
    block.close();
    checkCommon(block, initial, dependencies.get());

    block.setNotFastClonable(edm::FileBlock::NoEventsInFile);
    block.setNotFastClonable(edm::FileBlock::FileTooOld);
    block.setNotFastClonable(edm::FileBlock::NoEventsInFile);
    checkCommon(block, initial | edm::FileBlock::NoEventsInFile | edm::FileBlock::FileTooOld, dependencies.get());
  }
}

TEST_CASE("FileBlock shares dependency contents and lifetime", "[FileBlock]") {
  auto dependencies = std::make_shared<edm::ProductDependencies>();
  dependencies->insertChild(edm::BranchID(11), edm::BranchID(22));
  dependencies->insertEmpty(edm::BranchID(33));
  std::weak_ptr<edm::ProductDependencies> weak = dependencies;
  auto const* address = dependencies.get();
  {
    auto block = makeBlock(0, dependencies);
    {
      auto other = makeBlock(0, dependencies);
      dependencies.reset();
      REQUIRE_FALSE(weak.expired());
      CHECK(&block.productDependencies() == address);
      CHECK(&other.productDependencies() == address);
      auto const& lookup = other.productDependencies().childLookup();
      REQUIRE(lookup.size() == 2);
      auto parent = lookup.find(edm::BranchID(11));
      REQUIRE(parent != lookup.end());
      REQUIRE(parent->second.size() == 1);
      CHECK(parent->second.count(edm::BranchID(22)) == 1);
      auto emptyParent = lookup.find(edm::BranchID(33));
      REQUIRE(emptyParent != lookup.end());
      CHECK(emptyParent->second.empty());
    }
    CHECK_FALSE(weak.expired());
    CHECK(&block.productDependencies() == address);
  }
  CHECK(weak.expired());
}
