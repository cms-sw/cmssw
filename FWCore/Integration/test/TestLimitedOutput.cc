
#include "FWCore/Framework/interface/limited/OutputModule.h"
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

  class TestLimitedOutput : public limited::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestLimitedOutput(ParameterSet const& pset);
    ~TestLimitedOutput() override;
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

  TestLimitedOutput::TestLimitedOutput(ParameterSet const& pset)
      : limited::OutputModuleBase(pset),
        limited::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>>(pset),
        verbose_(pset.getUntrackedParameter<bool>("verbose")),
        expectedProcessesWithProcessBlockProducts_(
            pset.getUntrackedParameter<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts")),
        expectedWriteProcessBlockTransitions_(pset.getUntrackedParameter<int>("expectedWriteProcessBlockTransitions")) {
  }

  TestLimitedOutput::~TestLimitedOutput() {}

  void TestLimitedOutput::write(EventForOutput const& e) {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited write event";
    }
  }

  void TestLimitedOutput::writeLuminosityBlock(LuminosityBlockForOutput const&) {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited writeLuminosityBlock";
    }
  }

  void TestLimitedOutput::writeRun(RunForOutput const&) { LogAbsolute("TestLimitedOutput") << "limited writeRun"; }

  void TestLimitedOutput::writeProcessBlock(ProcessBlockForOutput const&) {
    LogAbsolute("TestLimitedOutput") << "limited writeProcessBlock";
    ++countWriteProcessBlockTransitions_;
    if (!expectedProcessesWithProcessBlockProducts_.empty()) {
      for (auto const& process : outputProcessBlockHelper().processesWithProcessBlockProducts()) {
        LogAbsolute("TestLimitedOutput") << "    " << process;
      }
      if (expectedProcessesWithProcessBlockProducts_ !=
          outputProcessBlockHelper().processesWithProcessBlockProducts()) {
        throw cms::Exception("TestFailure") << "TestLimitedOutput::writeProcessBlock unexpected process name list";
      }
    }
  }

  void TestLimitedOutput::respondToOpenInputFile(FileBlock const&) {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited respondToOpenInputFile";
    }
  }

  void TestLimitedOutput::respondToCloseInputFile(FileBlock const&) {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited respondToCloseInputFile";
    }
  }

  std::shared_ptr<int> TestLimitedOutput::globalBeginRun(RunForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalBeginRun";
    if (verbose_) {
      BranchIDLists const* theBranchIDLists = branchIDLists();
      for (auto const& branchIDList : *theBranchIDLists) {
        LogAbsolute("TestLimitedOutput") << "A branchID list";
        for (auto const& branchID : branchIDList) {
          LogAbsolute("TestLimitedOutput") << "  limited branchID " << branchID;
        }
      }
      edm::Service<edm::ConstProductRegistry> reg;
      for (auto const& it : reg->productList()) {
        LogAbsolute("TestLimitedOutput") << it.second;
      }
    }
    return std::make_shared<int>(0);
  }

  void TestLimitedOutput::globalEndRun(RunForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalEndRun";
  }

  std::shared_ptr<int> TestLimitedOutput::globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited globalBeginLuminosityBlock";
    }
    return std::make_shared<int>(0);
  }

  void TestLimitedOutput::globalEndLuminosityBlock(LuminosityBlockForOutput const&) const {
    if (verbose_) {
      LogAbsolute("TestLimitedOutput") << "limited globalEndLuminosityBlock";
    }
  }

  void TestLimitedOutput::endJob() {
    if (expectedWriteProcessBlockTransitions_ >= 0) {
      if (expectedWriteProcessBlockTransitions_ != countWriteProcessBlockTransitions_) {
        throw cms::Exception("TestFailure")
            << "TestLimitedOutput::writeProcessBlock unexpected number of writeProcessBlock transitions";
      }
    }
  }

  void TestLimitedOutput::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    desc.addUntracked<bool>("verbose", true);
    desc.addUntracked<std::vector<std::string>>("expectedProcessesWithProcessBlockProducts",
                                                std::vector<std::string>());
    desc.addUntracked<int>("expectedWriteProcessBlockTransitions", -1);
    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::TestLimitedOutput;
DEFINE_FWK_MODULE(TestLimitedOutput);
