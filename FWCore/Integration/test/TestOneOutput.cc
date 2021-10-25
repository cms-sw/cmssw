
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TTree.h"

#include <memory>
#include <string>
#include <vector>

constexpr unsigned int kDoNotTest = 0xffffffff;

namespace edm {

  class TestOneOutput : public one::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestOneOutput(ParameterSet const& pset);
    ~TestOneOutput() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;
    void writeProcessBlock(ProcessBlockForOutput const&) override;

    void respondToOpenInputFile(FileBlock const&) override;
    void respondToCloseInputFile(FileBlock const&) override;
    void testFileBlock(FileBlock const&);

    std::shared_ptr<int> globalBeginRun(RunForOutput const&) const override;
    void globalEndRun(RunForOutput const&) override;

    std::shared_ptr<int> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void globalEndLuminosityBlock(LuminosityBlockForOutput const&) override;
    void endJob() override;

    bool verbose_;
    std::vector<std::string> expectedProcessesWithProcessBlockProducts_;
    std::vector<std::string> expectedTopProcessesWithProcessBlockProducts_;
    std::vector<std::string> expectedAddedProcesses_;
    std::vector<std::string> expectedTopAddedProcesses_;
    std::vector<unsigned int> expectedTopCacheIndices0_;
    std::vector<unsigned int> expectedTopCacheIndices1_;
    std::vector<unsigned int> expectedTopCacheIndices2_;
    std::vector<std::string> expectedProcessNamesAtWrite_;
    int expectedWriteProcessBlockTransitions_;
    unsigned int countWriteProcessBlockTransitions_ = 0;
    unsigned int countInputFiles_ = 0;
    bool requireNullTTreesInFileBlock_;
    bool testTTreesInFileBlock_;
    std::vector<unsigned int> expectedCacheIndexSize_;
    unsigned int expectedProcessesInFirstFile_;
    std::vector<unsigned int> expectedCacheIndexVectorsPerFile_;
    std::vector<unsigned int> expectedNEntries0_;
    std::vector<unsigned int> expectedNEntries1_;
    std::vector<unsigned int> expectedNEntries2_;
    std::vector<unsigned int> expectedCacheEntriesPerFile0_;
    std::vector<unsigned int> expectedCacheEntriesPerFile1_;
    std::vector<unsigned int> expectedCacheEntriesPerFile2_;
    std::vector<unsigned int> expectedOuterOffset_;
    std::vector<unsigned int> expectedTranslateFromStoredIndex_;
    unsigned int expectedNAddedProcesses_;
    bool expectedProductsFromInputKept_;
  };

  TestOneOutput::TestOneOutput(ParameterSet const& pset)
      : one::OutputModuleBase(pset),
        one::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>>(pset),
        verbose_(pset.getUntrackedParameter<bool>("verbose")),
        expectedProcessesWithProcessBlockProducts_(
            pset.getUntrackedParameter<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts")),
        expectedTopProcessesWithProcessBlockProducts_(
            pset.getUntrackedParameter<std::vector<std::string>>("expectedTopProcessesWithProcessBlockProducts")),
        expectedAddedProcesses_(pset.getUntrackedParameter<std::vector<std::string>>("expectedAddedProcesses")),
        expectedTopAddedProcesses_(pset.getUntrackedParameter<std::vector<std::string>>("expectedTopAddedProcesses")),
        expectedTopCacheIndices0_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedTopCacheIndices0")),
        expectedTopCacheIndices1_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedTopCacheIndices1")),
        expectedTopCacheIndices2_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedTopCacheIndices2")),
        expectedProcessNamesAtWrite_(
            pset.getUntrackedParameter<std::vector<std::string>>("expectedProcessNamesAtWrite")),
        expectedWriteProcessBlockTransitions_(pset.getUntrackedParameter<int>("expectedWriteProcessBlockTransitions")),
        requireNullTTreesInFileBlock_(pset.getUntrackedParameter<bool>("requireNullTTreesInFileBlock")),
        testTTreesInFileBlock_(pset.getUntrackedParameter<bool>("testTTreesInFileBlock")),
        expectedCacheIndexSize_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheIndexSize")),
        expectedProcessesInFirstFile_(pset.getUntrackedParameter<unsigned int>("expectedProcessesInFirstFile")),
        expectedCacheIndexVectorsPerFile_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheIndexVectorsPerFile")),
        expectedNEntries0_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedNEntries0")),
        expectedNEntries1_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedNEntries1")),
        expectedNEntries2_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedNEntries2")),
        expectedCacheEntriesPerFile0_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheEntriesPerFile0")),
        expectedCacheEntriesPerFile1_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheEntriesPerFile1")),
        expectedCacheEntriesPerFile2_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheEntriesPerFile2")),
        expectedOuterOffset_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedOuterOffset")),
        expectedTranslateFromStoredIndex_(
            pset.getUntrackedParameter<std::vector<unsigned int>>("expectedTranslateFromStoredIndex")),
        expectedNAddedProcesses_(pset.getUntrackedParameter<unsigned int>("expectedNAddedProcesses")),
        expectedProductsFromInputKept_(pset.getUntrackedParameter<bool>("expectedProductsFromInputKept")) {}

  TestOneOutput::~TestOneOutput() {}

  void TestOneOutput::write(EventForOutput const& e) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one write event";
    }
  }

  void TestOneOutput::writeLuminosityBlock(LuminosityBlockForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one writeLuminosityBlock";
    }
  }

  void TestOneOutput::writeRun(RunForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one writeRun";
    }
  }

  void TestOneOutput::writeProcessBlock(ProcessBlockForOutput const& pb) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one writeProcessBlock";
    }
    if (countWriteProcessBlockTransitions_ < expectedProcessNamesAtWrite_.size()) {
      if (expectedProcessNamesAtWrite_[countWriteProcessBlockTransitions_] != pb.processName()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected process name";
      }
    }
    if (!expectedProcessesWithProcessBlockProducts_.empty()) {
      if (verbose_) {
        for (auto const& process : outputProcessBlockHelper().processesWithProcessBlockProducts()) {
          LogAbsolute("TestOneOutput") << "    " << process;
        }
      }
      if (expectedProcessesWithProcessBlockProducts_ !=
          outputProcessBlockHelper().processesWithProcessBlockProducts()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected process name list";
      }
    }
    if (!(!expectedAddedProcesses_.empty() && expectedAddedProcesses_[0] == "DONOTTEST")) {
      if (expectedAddedProcesses_ != outputProcessBlockHelper().processBlockHelper()->addedProcesses()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected addedProcesses list";
      }
    }
    if (!(!expectedTopAddedProcesses_.empty() && expectedTopAddedProcesses_[0] == "DONOTTEST")) {
      if (expectedTopAddedProcesses_ !=
          outputProcessBlockHelper().processBlockHelper()->topProcessBlockHelper()->addedProcesses()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected top addedProcesses list";
      }
    }
    if (!expectedTopProcessesWithProcessBlockProducts_.empty()) {
      // Same test as the previous except check the list of names in
      // the top level process name list from the EventProcessor
      if (expectedTopProcessesWithProcessBlockProducts_ != outputProcessBlockHelper()
                                                               .processBlockHelper()
                                                               ->topProcessBlockHelper()
                                                               ->topProcessesWithProcessBlockProducts()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected top process name list";
      }
      // Almost the same as the previous test, should get the same result
      if (expectedTopProcessesWithProcessBlockProducts_ !=
          outputProcessBlockHelper().processBlockHelper()->topProcessesWithProcessBlockProducts()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected top process name list 2";
      }

      std::vector<unsigned int> const* expectedTopCacheIndices = nullptr;
      if (countInputFiles_ == 1 && !expectedTopCacheIndices0_.empty()) {
        expectedTopCacheIndices = &expectedTopCacheIndices0_;
      } else if (countInputFiles_ == 2 && !expectedTopCacheIndices1_.empty()) {
        expectedTopCacheIndices = &expectedTopCacheIndices1_;
      } else if (countInputFiles_ == 3 && !expectedTopCacheIndices2_.empty()) {
        expectedTopCacheIndices = &expectedTopCacheIndices2_;
      }
      if (expectedTopCacheIndices != nullptr) {
        unsigned int expectedInputProcesses =
            expectedTopProcessesWithProcessBlockProducts_.size() -
            outputProcessBlockHelper().processBlockHelper()->topProcessBlockHelper()->addedProcesses().size();

        std::vector<std::vector<unsigned int>> const& topProcessBlockCacheIndices =
            outputProcessBlockHelper().processBlockHelper()->processBlockCacheIndices();
        if (expectedTopCacheIndices->size() != expectedInputProcesses * topProcessBlockCacheIndices.size()) {
          throw cms::Exception("TestFailure")
              << "TestOneOutput::writeProcessBlock unexpected sizes related to top cache indices on input file "
              << (countInputFiles_ - 1);
        }
        unsigned int iStored = 0;
        for (unsigned int i = 0; i < topProcessBlockCacheIndices.size(); ++i) {
          if (topProcessBlockCacheIndices[i].size() != expectedInputProcesses) {
            throw cms::Exception("TestFailure")
                << "TestOneOutput::writeProcessBlock unexpected size of inner cache indices vector on input file "
                << (countInputFiles_ - 1);
          }
          for (unsigned int j = 0; j < topProcessBlockCacheIndices[i].size(); ++j) {
            if (topProcessBlockCacheIndices[i][j] != (*expectedTopCacheIndices)[iStored]) {
              throw cms::Exception("TestFailure")
                  << "TestOneOutput::writeProcessBlock unexpected cache index value on input file "
                  << (countInputFiles_ - 1);
            }
            ++iStored;
          }
        }
      }
    }
    if (expectedProcessesInFirstFile_ != 0) {
      if (expectedProcessesInFirstFile_ != outputProcessBlockHelper().processBlockHelper()->nProcessesInFirstFile()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for nProcessesInFirstFile";
      }
    }
    for (unsigned int i = 0; i < expectedCacheIndexVectorsPerFile_.size(); ++i) {
      if (i < countInputFiles_) {
        if (expectedCacheIndexVectorsPerFile_[i] !=
            outputProcessBlockHelper().processBlockHelper()->cacheIndexVectorsPerFile()[i]) {
          throw cms::Exception("TestFailure")
              << "TestOneOutput::writeProcessBlock unexpected value for cacheIndexVectorsPerFile, element " << i;
        }
      }
    }
    if (countInputFiles_ >= 1 && !expectedNEntries0_.empty()) {
      if (expectedNEntries0_ != outputProcessBlockHelper().processBlockHelper()->nEntries()[0]) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected value for nEntries 0";
      }
    }
    if (countInputFiles_ >= 2 && !expectedNEntries1_.empty()) {
      if (expectedNEntries1_ != outputProcessBlockHelper().processBlockHelper()->nEntries()[1]) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected value for nEntries 1";
      }
    }
    if (countInputFiles_ >= 3 && !expectedNEntries2_.empty()) {
      if (expectedNEntries2_ != outputProcessBlockHelper().processBlockHelper()->nEntries()[2]) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected value for nEntries 2";
      }
    }

    if (countInputFiles_ == 1 && !expectedCacheEntriesPerFile0_.empty()) {
      if (expectedCacheEntriesPerFile0_ != outputProcessBlockHelper().processBlockHelper()->cacheEntriesPerFile()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for cacheEntriesPerFile 0";
      }
    } else if (countInputFiles_ == 2 && !expectedCacheEntriesPerFile1_.empty()) {
      if (expectedCacheEntriesPerFile1_ != outputProcessBlockHelper().processBlockHelper()->cacheEntriesPerFile()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for cacheEntriesPerFile 1";
      }
    } else if (countInputFiles_ == 3 && !expectedCacheEntriesPerFile2_.empty()) {
      if (expectedCacheEntriesPerFile2_ != outputProcessBlockHelper().processBlockHelper()->cacheEntriesPerFile()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for cacheEntriesPerFile 2";
      }
    }

    if (!expectedOuterOffset_.empty() && (countInputFiles_ - 1) < expectedOuterOffset_.size()) {
      if (expectedOuterOffset_[countInputFiles_ - 1] !=
          outputProcessBlockHelper().processBlockHelper()->outerOffset()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for outerOffset, file " << (countInputFiles_ - 1);
      }
    }
    if (countWriteProcessBlockTransitions_ < expectedCacheIndexSize_.size()) {
      if (expectedCacheIndexSize_[countWriteProcessBlockTransitions_] !=
          outputProcessBlockHelper().processBlockHelper()->processBlockCacheIndices().size()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected cache index size "
            << outputProcessBlockHelper().processBlockHelper()->processBlockCacheIndices().size();
      }
    }
    if (!expectedTranslateFromStoredIndex_.empty()) {
      if (expectedTranslateFromStoredIndex_ != outputProcessBlockHelper().translateFromStoredIndex()) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected value for translateFromStoredIndex";
      }
    }
    if (expectedNAddedProcesses_ != kDoNotTest) {
      if (expectedNAddedProcesses_ != outputProcessBlockHelper().nAddedProcesses()) {
        throw cms::Exception("TestFailure") << "TestOneOutput::writeProcessBlock unexpected value for nAddedProcesses";
      }
    }
    if (expectedProductsFromInputKept_ != outputProcessBlockHelper().productsFromInputKept()) {
      throw cms::Exception("TestFailure")
          << "TestOneOutput::writeProcessBlock unexpected value for productsFromInputKept";
    }
    ++countWriteProcessBlockTransitions_;
  }

  void TestOneOutput::respondToOpenInputFile(FileBlock const& fb) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one respondToOpenInputFile";
    }
    testFileBlock(fb);
    ++countInputFiles_;
  }

  void TestOneOutput::respondToCloseInputFile(FileBlock const& fb) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one respondToCloseInputFile";
    }
    testFileBlock(fb);
  }

  void TestOneOutput::testFileBlock(FileBlock const& fb) {
    if (requireNullTTreesInFileBlock_) {
      if (fb.tree() != nullptr || fb.lumiTree() != nullptr || fb.runTree() != nullptr ||
          !fb.processBlockTrees().empty() || !fb.processesWithProcessBlockTrees().empty() ||
          fb.processBlockTree(std::string("DoesNotExist")) != nullptr) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::respondToOpenInputFile expected null TTree pointers in FileBlock";
      }
    } else if (testTTreesInFileBlock_) {
      if (fb.tree() == nullptr || std::string("Events") != fb.tree()->GetName() ||
          std::string("LuminosityBlocks") != fb.lumiTree()->GetName() ||
          std::string("Runs") != fb.runTree()->GetName() || fb.processesWithProcessBlockTrees().size() != 2 ||
          fb.processesWithProcessBlockTrees()[0] != "PROD1" || fb.processesWithProcessBlockTrees()[1] != "MERGE" ||
          fb.processBlockTrees().size() != 2 ||
          std::string("ProcessBlocksPROD1") != fb.processBlockTrees()[0]->GetName() ||
          std::string("ProcessBlocksMERGE") != fb.processBlockTrees()[1]->GetName() ||
          std::string("ProcessBlocksPROD1") != fb.processBlockTree("PROD1")->GetName() ||
          std::string("ProcessBlocksMERGE") != fb.processBlockTree("MERGE")->GetName() ||
          fb.processBlockTree("DOESNOTEXIST") != nullptr) {
        throw cms::Exception("TestFailure") << "TestOneOutput::testFileBlock failed. Testing tree pointers";
      }
    }
  }

  std::shared_ptr<int> TestOneOutput::globalBeginRun(RunForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one globalBeginRun";
      edm::Service<edm::ConstProductRegistry> reg;
      for (auto const& it : reg->productList()) {
        LogAbsolute("TestOneOutput") << it.second;
      }
    }
    return std::make_shared<int>(0);
  }

  void TestOneOutput::globalEndRun(RunForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one globalEndRun";
    }
  }

  std::shared_ptr<int> TestOneOutput::globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one globalBeginLuminosityBlock";
    }
    return std::make_shared<int>(0);
  }

  void TestOneOutput::globalEndLuminosityBlock(LuminosityBlockForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestOneOutput") << "one globalEndLuminosityBlock";
    }
  }

  void TestOneOutput::endJob() {
    if (expectedWriteProcessBlockTransitions_ >= 0) {
      if (static_cast<unsigned int>(expectedWriteProcessBlockTransitions_) != countWriteProcessBlockTransitions_) {
        throw cms::Exception("TestFailure")
            << "TestOneOutput::writeProcessBlock unexpected number of writeProcessBlock transitions";
      }
    }
  }

  void TestOneOutput::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    desc.addUntracked<bool>("verbose", true);
    desc.addUntracked<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts",
                                                std::vector<std::string>());
    desc.addUntracked<std::vector<std::string>>("expectedTopProcessesWithProcessBlockProducts",
                                                std::vector<std::string>());
    desc.addUntracked<std::vector<std::string>>("expectedAddedProcesses",
                                                std::vector<std::string>(1, std::string("DONOTTEST")));
    desc.addUntracked<std::vector<std::string>>("expectedTopAddedProcesses",
                                                std::vector<std::string>(1, std::string("DONOTTEST")));
    desc.addUntracked<std::vector<unsigned int>>("expectedTopCacheIndices0", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedTopCacheIndices1", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedTopCacheIndices2", std::vector<unsigned int>());
    desc.addUntracked<std::vector<std::string>>("expectedProcessNamesAtWrite", std::vector<std::string>());
    desc.addUntracked<int>("expectedWriteProcessBlockTransitions", -1);
    desc.addUntracked<bool>("requireNullTTreesInFileBlock", false);
    desc.addUntracked<bool>("testTTreesInFileBlock", false);
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheIndexSize", std::vector<unsigned int>());
    desc.addUntracked<unsigned int>("expectedProcessesInFirstFile", 0);
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheIndexVectorsPerFile", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedNEntries0", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedNEntries1", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedNEntries2", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheEntriesPerFile0", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheEntriesPerFile1", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheEntriesPerFile2", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedOuterOffset", std::vector<unsigned int>());
    desc.addUntracked<std::vector<unsigned int>>("expectedTranslateFromStoredIndex", std::vector<unsigned int>());
    desc.addUntracked<unsigned int>("expectedNAddedProcesses", kDoNotTest);
    desc.addUntracked<bool>("expectedProductsFromInputKept", true);

    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::TestOneOutput;
DEFINE_FWK_MODULE(TestOneOutput);
