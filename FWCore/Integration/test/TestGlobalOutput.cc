
#include "FWCore/Framework/interface/global/OutputModule.h"
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

  class TestGlobalOutput : public global::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestGlobalOutput(ParameterSet const& pset);
    ~TestGlobalOutput() override;
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

  TestGlobalOutput::TestGlobalOutput(ParameterSet const& pset)
      : global::OutputModuleBase(pset),
        global::OutputModule<WatchInputFiles, RunCache<int>, LuminosityBlockCache<int>>(pset) {}

  TestGlobalOutput::~TestGlobalOutput() {}

  void TestGlobalOutput::write(EventForOutput const& e) { LogAbsolute("TestGlobalOutput") << "global write event"; }

  void TestGlobalOutput::writeLuminosityBlock(LuminosityBlockForOutput const&) {
    LogAbsolute("TestGlobalOutput") << "global writeLuminosityBlock";
  }

  void TestGlobalOutput::writeRun(RunForOutput const&) { LogAbsolute("TestGlobalOutput") << "global writeRun"; }

  void TestGlobalOutput::respondToOpenInputFile(FileBlock const&) {
    LogAbsolute("TestGlobalOutput") << "global respondToOpenInputFile";
  }

  void TestGlobalOutput::respondToCloseInputFile(FileBlock const&) {
    LogAbsolute("TestGlobalOutput") << "global respondToCloseInputFile";
  }

  std::shared_ptr<int> TestGlobalOutput::globalBeginRun(RunForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalBeginRun";
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
    return std::make_shared<int>(0);
  }

  void TestGlobalOutput::globalEndRun(RunForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalEndRun";
  }

  std::shared_ptr<int> TestGlobalOutput::globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalBeginLuminosityBlock";
    return std::make_shared<int>(0);
  }

  void TestGlobalOutput::globalEndLuminosityBlock(LuminosityBlockForOutput const&) const {
    LogAbsolute("TestGlobalOutput") << "global globalEndLuminosityBlock";
  }

  void TestGlobalOutput::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    descriptions.addDefault(desc);
  }
}  // namespace edm

using edm::TestGlobalOutput;
DEFINE_FWK_MODULE(TestGlobalOutput);
