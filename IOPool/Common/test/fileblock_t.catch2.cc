#include "catch2/catch_all.hpp"

#include "FWCore/Framework/interface/FileBlock.h"
#include "IOPool/Common/interface/TTreeFileBlock.h"
#include "DataFormats/Provenance/interface/BranchID.h"

#include "TTree.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
  static_assert(!std::is_copy_constructible_v<edm::TTreeFileBlock>);
  static_assert(!std::is_copy_assignable_v<edm::TTreeFileBlock>);
  static_assert(!std::is_move_constructible_v<edm::TTreeFileBlock>);
  static_assert(!std::is_move_assignable_v<edm::TTreeFileBlock>);

  using Trees = std::array<TTree*, 6>;

  struct Fixture {
    Fixture() : dependencies(std::make_shared<edm::ProductDependencies>()) {
      for (unsigned int i = 0; i != trees.size(); ++i) {
        auto name = std::string("fileBlockTree") + std::to_string(i);
        trees[i] = std::make_unique<TTree>(name.c_str(), name.c_str());
        trees[i]->SetDirectory(nullptr);
      }
      dependencies->insertChild(edm::BranchID(11), edm::BranchID(22));
      dependencies->insertEmpty(edm::BranchID(33));
    }

    Trees firstTrees() const {
      return {trees[0].get(), trees[1].get(), trees[2].get(), trees[3].get(), trees[4].get(), trees[5].get()};
    }

    Trees secondTrees() const {
      return {trees[5].get(), trees[4].get(), trees[3].get(), trees[2].get(), trees[1].get(), trees[0].get()};
    }

    edm::TTreeFileBlock makeBlock(int whyNotFastClonable) const {
      auto t = firstTrees();
      return edm::TTreeFileBlock(edm::FileFormatVersion(17),
                                 t[0],
                                 t[1],
                                 t[2],
                                 t[3],
                                 t[4],
                                 t[5],
                                 {trees[6].get(), nullptr, trees[7].get()},
                                 {"FIRST", "NULL", "LAST"},
                                 whyNotFastClonable,
                                 dropped,
                                 "input.root",
                                 true,
                                 true,
                                 dependencies);
    }

    std::shared_ptr<edm::TTreeFileBlock> makeSharedBlock(int whyNotFastClonable) const {
      return std::shared_ptr<edm::TTreeFileBlock>(new edm::TTreeFileBlock(makeBlock(whyNotFastClonable)));
    }

    std::array<std::unique_ptr<TTree>, 8> trees;
    std::shared_ptr<edm::ProductDependencies> dependencies;
    std::array<bool, edm::NumBranchTypes> dropped{{true, false, true, false}};
  };

  void checkTrees(edm::TTreeFileBlock const& block, Trees const& expected) {
    CHECK(block.tree() == expected[0]);
    CHECK(block.metaTree() == expected[1]);
    CHECK(block.lumiTree() == expected[2]);
    CHECK(block.lumiMetaTree() == expected[3]);
    CHECK(block.runTree() == expected[4]);
    CHECK(block.runMetaTree() == expected[5]);
  }

  void checkCommon(edm::FileBlock const& block, int whyNotFastClonable, edm::ProductDependencies const* dependencies) {
    CHECK(block.fileFormatVersion().value() == 17);
    CHECK(block.whyNotFastClonable() == whyNotFastClonable);
    CHECK(block.hasNewlyDroppedBranch() == std::array<bool, edm::NumBranchTypes>{{true, false, true, false}});
    CHECK(block.fileName() == "input.root");
    CHECK(block.branchListIndexesUnchanged());
    CHECK(block.modifiedIDs());
    CHECK(&block.productDependencies() == dependencies);
  }

  void refresh(edm::TTreeFileBlock& block,
               Trees const& trees,
               std::vector<TTree*> processTrees,
               std::vector<std::string> names) {
    block.updateTTreePointers(trees[0], trees[1], trees[2], trees[3], trees[4], trees[5], processTrees, names);
  }

}  // namespace

TEST_CASE("TTreeFileBlock defaults", "[TTreeFileBlock]") {
  edm::TTreeFileBlock treeBlock;
  CHECK(treeBlock.fileFormatVersion().value() == -1);
  CHECK(treeBlock.whyNotFastClonable() == edm::FileBlock::NoRootInputSource);
  for (auto dropped : treeBlock.hasNewlyDroppedBranch()) {
    CHECK_FALSE(dropped);
  }
  CHECK(treeBlock.fileName().empty());
  CHECK_FALSE(treeBlock.branchListIndexesUnchanged());
  CHECK_FALSE(treeBlock.modifiedIDs());
  CHECK(treeBlock.productDependencies().childLookup().empty());
  checkTrees(treeBlock, Trees{});
  CHECK(treeBlock.processBlockTrees().empty());
  CHECK(treeBlock.processesWithProcessBlockTrees().empty());
  CHECK(treeBlock.processBlockTree("missing") == nullptr);
}

TEST_CASE("TTreeFileBlock polymorphism", "[TTreeFileBlock]") {
  Fixture fixture;
  auto const reasons = edm::FileBlock::FileTooOld | edm::FileBlock::EventsToBeSorted;
  std::shared_ptr<edm::FileBlock> plain = std::make_shared<edm::FileBlock>();
  CHECK(std::dynamic_pointer_cast<edm::TTreeFileBlock>(plain) == nullptr);

  auto derived = fixture.makeSharedBlock(reasons);
  std::shared_ptr<edm::FileBlock> base = derived;
  auto cast = std::dynamic_pointer_cast<edm::TTreeFileBlock>(base);
  REQUIRE(cast);
  CHECK(cast == derived);
  CHECK_FALSE(cast.owner_before(base));
  CHECK_FALSE(base.owner_before(cast));
  checkCommon(*base, reasons, fixture.dependencies.get());
}

TEST_CASE("TTreeFileBlock construction and shared product dependencies", "[TTreeFileBlock]") {
  Fixture fixture;
  auto const reasons = edm::FileBlock::FileTooOld | edm::FileBlock::EventsToBeSorted;
  for (int whyNotFastClonable : {0, reasons}) {
    auto block = fixture.makeBlock(whyNotFastClonable);
    checkCommon(block, whyNotFastClonable, fixture.dependencies.get());
    checkTrees(block, fixture.firstTrees());
    CHECK(block.processBlockTrees() == std::vector<TTree*>{fixture.trees[6].get(), nullptr, fixture.trees[7].get()});
    CHECK(block.processesWithProcessBlockTrees() == std::vector<std::string>{"FIRST", "NULL", "LAST"});
    CHECK(block.processBlockTree("FIRST") == fixture.trees[6].get());
    CHECK(block.processBlockTree("NULL") == nullptr);
    CHECK(block.processBlockTree("LAST") == fixture.trees[7].get());
    CHECK(block.processBlockTree("missing") == nullptr);
  }

  auto block = fixture.makeBlock(0);
  auto const* address = fixture.dependencies.get();
  std::weak_ptr<edm::ProductDependencies> weak = fixture.dependencies;
  fixture.dependencies.reset();
  REQUIRE_FALSE(weak.expired());
  CHECK(&block.productDependencies() == address);
  auto const& lookup = block.productDependencies().childLookup();
  REQUIRE(lookup.size() == 2);
  auto parent = lookup.find(edm::BranchID(11));
  REQUIRE(parent != lookup.end());
  REQUIRE(parent->second.size() == 1);
  CHECK(parent->second.count(edm::BranchID(22)) == 1);
  auto emptyParent = lookup.find(edm::BranchID(33));
  REQUIRE(emptyParent != lookup.end());
  CHECK(emptyParent->second.empty());
}

TEST_CASE("TTreeFileBlock refresh and repeated close", "[TTreeFileBlock]") {
  Fixture fixture;
  auto const initialWhyNotFastClonable = edm::FileBlock::FileTooOld | edm::FileBlock::EventsToBeSorted;
  auto block = fixture.makeBlock(initialWhyNotFastClonable);
  auto const* dependencies = fixture.dependencies.get();
  edm::FileBlock* direct = &block;

  refresh(block, fixture.secondTrees(), {fixture.trees[7].get(), fixture.trees[6].get()}, {"LAST", "FIRST"});
  checkTrees(block, fixture.secondTrees());
  CHECK(block.processBlockTrees() == std::vector<TTree*>{fixture.trees[7].get(), fixture.trees[6].get()});
  CHECK(block.processesWithProcessBlockTrees() == std::vector<std::string>{"LAST", "FIRST"});
  CHECK(block.processBlockTree("LAST") == fixture.trees[7].get());
  CHECK(block.processBlockTree("FIRST") == fixture.trees[6].get());
  CHECK(block.processBlockTree("NULL") == nullptr);
  checkCommon(block, initialWhyNotFastClonable, dependencies);

  refresh(block, fixture.firstTrees(), {}, {});
  CHECK(block.processBlockTrees().empty());
  CHECK(block.processesWithProcessBlockTrees().empty());
  CHECK(block.processBlockTree("FIRST") == nullptr);
  checkTrees(block, fixture.firstTrees());
  checkCommon(block, initialWhyNotFastClonable, dependencies);
  direct->close();
  direct->close();
  checkTrees(block, Trees{});
  CHECK(block.processBlockTrees().empty());
  CHECK(block.processesWithProcessBlockTrees().empty());
  checkCommon(block, initialWhyNotFastClonable, dependencies);

  block.setNotFastClonable(edm::FileBlock::NoEventsInFile);
  block.setNotFastClonable(edm::FileBlock::FileTooOld);
  auto const finalWhyNotFastClonable = initialWhyNotFastClonable | edm::FileBlock::NoEventsInFile;
  checkCommon(block, finalWhyNotFastClonable, dependencies);
  refresh(block,
          fixture.secondTrees(),
          {fixture.trees[7].get(), nullptr, fixture.trees[6].get()},
          {"LAST", "NULL", "FIRST"});
  for (int i = 0; i != 2; ++i) {
    direct->close();
    checkTrees(block, Trees{});
    CHECK(block.processBlockTrees() == std::vector<TTree*>{nullptr, nullptr, nullptr});
    CHECK(block.processesWithProcessBlockTrees() == std::vector<std::string>{"LAST", "NULL", "FIRST"});
    CHECK(block.processBlockTree("LAST") == nullptr);
    CHECK(block.processBlockTree("NULL") == nullptr);
    CHECK(block.processBlockTree("FIRST") == nullptr);
    CHECK(block.processBlockTree("missing") == nullptr);
    checkCommon(block, finalWhyNotFastClonable, dependencies);
  }
}

TEST_CASE("TTreeFileBlock does not own trees", "[TTreeFileBlock]") {
  Fixture fixture;
  {
    auto closed = fixture.makeBlock(0);
    closed.close();
    auto destroyedWithLivePointers = fixture.makeBlock(0);
  }
  for (unsigned int i = 0; i != fixture.trees.size(); ++i) {
    CHECK(fixture.trees[i]->GetDirectory() == nullptr);
    CHECK(std::string(fixture.trees[i]->GetName()) == std::string("fileBlockTree") + std::to_string(i));
    CHECK(fixture.trees[i]->GetEntries() == 0);
  }
}
