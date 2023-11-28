// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      TestOutputWithGetterOfProductsGlobal
//
/**\class edm::TestOutputWithGetterOfProductsGlobal

  Description: Test GetterOfProducts with OutputModule
*/
// Original Author:  W. David Dagenhart
//         Created:  26 April 2023

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/OutputModule.h"
#include "FWCore/Framework/interface/TypeMatch.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>
#include <vector>

namespace edm {

  class TestOutputWithGetterOfProductsGlobal : public global::OutputModule<RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestOutputWithGetterOfProductsGlobal(ParameterSet const&);
    static void fillDescriptions(ConfigurationDescriptions&);

  private:
    void write(EventForOutput const&) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;

    std::shared_ptr<int> globalBeginRun(RunForOutput const&) const override;
    void globalEndRun(RunForOutput const&) const override;

    std::shared_ptr<int> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void globalEndLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void endJob() override;

    int sumThings(std::vector<Handle<edmtest::ThingCollection>> const& collections) const;

    mutable std::atomic<unsigned int> sum_ = 0;
    unsigned int expectedSum_;
    GetterOfProducts<edmtest::ThingCollection> getterOfProductsRun_;
    GetterOfProducts<edmtest::ThingCollection> getterOfProductsLumi_;
    GetterOfProducts<edmtest::ThingCollection> getterOfProductsEvent_;
  };

  TestOutputWithGetterOfProductsGlobal::TestOutputWithGetterOfProductsGlobal(ParameterSet const& pset)
      : global::OutputModuleBase(pset),
        global::OutputModule<RunCache<int>, LuminosityBlockCache<int>>(pset),
        expectedSum_(pset.getUntrackedParameter<unsigned int>("expectedSum")),
        getterOfProductsRun_(TypeMatch(), this, InRun),
        getterOfProductsLumi_(edm::TypeMatch(), this, InLumi),
        getterOfProductsEvent_(edm::TypeMatch(), this) {
    callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
      getterOfProductsRun_(bd);
      getterOfProductsLumi_(bd);
      getterOfProductsEvent_(bd);
    });
  }

  void TestOutputWithGetterOfProductsGlobal::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    desc.addUntracked<unsigned int>("expectedSum", 0);
    descriptions.addDefault(desc);
  }

  void TestOutputWithGetterOfProductsGlobal::write(EventForOutput const& event) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEvent_.fillHandles(event, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProductsGlobal::writeLuminosityBlock(LuminosityBlockForOutput const& lumi) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsLumi_.fillHandles(lumi, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProductsGlobal::writeRun(RunForOutput const& run) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsRun_.fillHandles(run, handles);
    sum_ += sumThings(handles);
  }

  std::shared_ptr<int> TestOutputWithGetterOfProductsGlobal::globalBeginRun(RunForOutput const& run) const {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsRun_.fillHandles(run, handles);
    return std::make_shared<int>(sumThings(handles));
  }

  void TestOutputWithGetterOfProductsGlobal::globalEndRun(RunForOutput const& run) const {
    sum_ += *runCache(run.index());
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsRun_.fillHandles(run, handles);
    sum_ += sumThings(handles);
  }

  std::shared_ptr<int> TestOutputWithGetterOfProductsGlobal::globalBeginLuminosityBlock(
      LuminosityBlockForOutput const& lumi) const {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsLumi_.fillHandles(lumi, handles);
    return std::make_shared<int>(sumThings(handles));
  }

  void TestOutputWithGetterOfProductsGlobal::globalEndLuminosityBlock(LuminosityBlockForOutput const& lumi) const {
    sum_ += *luminosityBlockCache(lumi.index());
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsLumi_.fillHandles(lumi, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProductsGlobal::endJob() {
    if (expectedSum_ != 0 && expectedSum_ != sum_.load()) {
      throw cms::Exception("TestFailure") << "TestOutputWithGetterOfProductsGlobal::endJob, sum = " << sum_.load()
                                          << " which is not equal to the expected value " << expectedSum_;
    }
  }

  int TestOutputWithGetterOfProductsGlobal::sumThings(
      std::vector<Handle<edmtest::ThingCollection>> const& collections) const {
    int sum = 0;
    for (auto const& collection : collections) {
      for (auto const& thing : *collection) {
        sum += thing.a;
      }
    }
    return sum;
  }
}  // namespace edm

using edm::TestOutputWithGetterOfProductsGlobal;
DEFINE_FWK_MODULE(TestOutputWithGetterOfProductsGlobal);
