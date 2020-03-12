
#include "FWCore/Framework/interface/limited/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
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

    void respondToOpenInputFile(FileBlock const&) override;
    void respondToCloseInputFile(FileBlock const&) override;

    std::shared_ptr<int> globalBeginRun(RunForOutput const&) const override;
    void globalEndRun(RunForOutput const&) const override;

    std::shared_ptr<int> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void globalEndLuminosityBlock(LuminosityBlockForOutput const&) const override;
  };

  TestLimitedOutput::TestLimitedOutput(ParameterSet const& pset)
      : limited::OutputModuleBase(pset),
        limited::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>>(pset) {}

  TestLimitedOutput::~TestLimitedOutput() {}

  void TestLimitedOutput::write(EventForOutput const& e) { LogAbsolute("TestLimitedOutput") << "limited write event"; }

  void TestLimitedOutput::writeLuminosityBlock(LuminosityBlockForOutput const&) {
    LogAbsolute("TestLimitedOutput") << "limited writeLuminosityBlock";
  }

  void TestLimitedOutput::writeRun(RunForOutput const&) { LogAbsolute("TestLimitedOutput") << "limited writeRun"; }

  void TestLimitedOutput::respondToOpenInputFile(FileBlock const&) {
    LogAbsolute("TestLimitedOutput") << "limited respondToOpenInputFile";
  }

  void TestLimitedOutput::respondToCloseInputFile(FileBlock const&) {
    LogAbsolute("TestLimitedOutput") << "limited respondToCloseInputFile";
  }

  std::shared_ptr<int> TestLimitedOutput::globalBeginRun(RunForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalBeginRun";
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
    return std::make_shared<int>(0);
  }

  void TestLimitedOutput::globalEndRun(RunForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalEndRun";
  }

  std::shared_ptr<int> TestLimitedOutput::globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalBeginLuminosityBlock";
    return std::make_shared<int>(0);
  }

  void TestLimitedOutput::globalEndLuminosityBlock(LuminosityBlockForOutput const&) const {
    LogAbsolute("TestLimitedOutput") << "limited globalEndLuminosityBlock";
  }

  void TestLimitedOutput::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::TestLimitedOutput;
DEFINE_FWK_MODULE(TestLimitedOutput);
