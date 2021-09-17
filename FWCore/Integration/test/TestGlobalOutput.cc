
#include "FWCore/Framework/interface/global/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {

  class TestGlobalOutput : public global::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestGlobalOutput(ParameterSet const& pset);
    ~TestGlobalOutput() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;
    void writeProcessBlock(ProcessBlockForOutput const&) override;

    void respondToOpenInputFile(FileBlock const&) override;
    void respondToCloseInputFile(FileBlock const&) override;

    std::shared_ptr<int> globalBeginRun(RunForOutput const&) const override;
    void globalEndRun(RunForOutput const&) const override;

    std::shared_ptr<int> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void globalEndLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void endJob() override;

    bool verbose_;
    std::vector<std::string> expectedProcessesWithProcessBlockProducts_;
    int expectedWriteProcessBlockTransitions_;
    int countWriteProcessBlockTransitions_ = 0;
  };

  TestGlobalOutput::TestGlobalOutput(ParameterSet const& pset)
      : global::OutputModuleBase(pset),
        global::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>>(pset),
        verbose_(pset.getUntrackedParameter<bool>("verbose")),
        expectedProcessesWithProcessBlockProducts_(
            pset.getUntrackedParameter<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts")),
        expectedWriteProcessBlockTransitions_(pset.getUntrackedParameter<int>("expectedWriteProcessBlockTransitions")) {
  }

  TestGlobalOutput::~TestGlobalOutput() {}

  void TestGlobalOutput::write(EventForOutput const& e) {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global write event";
    }
  }

  void TestGlobalOutput::writeLuminosityBlock(LuminosityBlockForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global writeLuminosityBlock";
    }
  }

  void TestGlobalOutput::writeRun(RunForOutput const&) { LogAbsolute("TestGlobalOutput") << "global writeRun"; }

  void TestGlobalOutput::writeProcessBlock(ProcessBlockForOutput const&) {
    LogAbsolute("TestGlobalOutput") << "global writeProcessBlock";
    ++countWriteProcessBlockTransitions_;
    if (!expectedProcessesWithProcessBlockProducts_.empty()) {
      for (auto const& process : outputProcessBlockHelper().processesWithProcessBlockProducts()) {
        LogAbsolute("TestGlobalOutput") << "    " << process;
      }
      if (expectedProcessesWithProcessBlockProducts_ !=
          outputProcessBlockHelper().processesWithProcessBlockProducts()) {
        throw cms::Exception("TestFailure") << "TestGlobalOutput::writeProcessBlock unexpected process name list";
      }
    }
  }

  void TestGlobalOutput::respondToOpenInputFile(FileBlock const&) {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global respondToOpenInputFile";
    }
  }

  void TestGlobalOutput::respondToCloseInputFile(FileBlock const&) {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global respondToCloseInputFile";
    }
  }

  std::shared_ptr<int> TestGlobalOutput::globalBeginRun(RunForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalBeginRun";
    if (verbose_) {
      BranchIDLists const* theBranchIDLists = branchIDLists();
      for (auto const& branchIDList : *theBranchIDLists) {
        LogAbsolute("TestGlobalOutput") << "A branchID list";
        for (auto const& branchID : branchIDList) {
          LogAbsolute("TestGlobalOutput") << "  global branchID " << branchID;
        }
      }
      edm::Service<edm::ConstProductRegistry> reg;
      for (auto const& it : reg->productList()) {
        LogAbsolute("TestGlobalOutput") << it.second;
      }
    }
    return std::make_shared<int>(0);
  }

  void TestGlobalOutput::globalEndRun(RunForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalEndRun";
  }

  std::shared_ptr<int> TestGlobalOutput::globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global globalBeginLuminosityBlock";
    }
    return std::make_shared<int>(0);
  }

  void TestGlobalOutput::globalEndLuminosityBlock(LuminosityBlockForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestGlobalOutput") << "global globalEndLuminosityBlock";
    }
  }

  void TestGlobalOutput::endJob() {
    if (expectedWriteProcessBlockTransitions_ >= 0) {
      if (expectedWriteProcessBlockTransitions_ != countWriteProcessBlockTransitions_) {
        throw cms::Exception("TestFailure")
            << "TestGlobalOutput::writeProcessBlock unexpected number of writeProcessBlock transitions";
      }
    }
  }

  void TestGlobalOutput::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    desc.addUntracked<bool>("verbose", true);
    desc.addUntracked<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts",
                                                std::vector<std::string>());
    desc.addUntracked<int>("expectedWriteProcessBlockTransitions", -1);
    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::TestGlobalOutput;
DEFINE_FWK_MODULE(TestGlobalOutput);
