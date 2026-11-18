// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      TestOutputWithGetterOfProducts
//
/**\class edm::TestOutputWithGetterOfProducts

  Description: Test GetterOfProducts with OutputModule
*/
// Original Author:  W. David Dagenhart
//         Created:  26 April 2023

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/TypeMatch.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

namespace edm {

  class TestOutputWithGetterOfProducts : public one::OutputModule<RunCache<int>, LuminosityBlockCache<int>> {
  public:
    explicit TestOutputWithGetterOfProducts(ParameterSet const&);
    static void fillDescriptions(ConfigurationDescriptions&);

  private:
    void write(EventForOutput const&) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;

    std::shared_ptr<int> globalBeginRun(RunForOutput const&) const override;
    void globalEndRun(RunForOutput const&) override;

    std::shared_ptr<int> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const override;
    void globalEndLuminosityBlock(LuminosityBlockForOutput const&) override;
    void endJob() override;

    int sumThings(std::vector<Handle<edmtest::ThingCollection>> const& collections) const;

    unsigned int sum_ = 0;
    unsigned int expectedSum_;
    GetterOfProducts<edmtest::ThingCollection, edm::Transition::BeginRun> getterOfProductsBeginRun_;
    GetterOfProducts<edmtest::ThingCollection, edm::Transition::EndRun> getterOfProductsEndRun_;
    GetterOfProducts<edmtest::ThingCollection, edm::Transition::BeginLuminosityBlock> getterOfProductsBeginLumi_;
    GetterOfProducts<edmtest::ThingCollection, edm::Transition::EndLuminosityBlock> getterOfProductsEndLumi_;
    GetterOfProducts<edmtest::ThingCollection> getterOfProductsEvent_;
  };

  TestOutputWithGetterOfProducts::TestOutputWithGetterOfProducts(ParameterSet const& pset)
      : one::OutputModuleBase(pset),
        one::OutputModule<RunCache<int>, LuminosityBlockCache<int>>(pset),
        expectedSum_(pset.getUntrackedParameter<unsigned int>("expectedSum")),
        getterOfProductsBeginRun_(TypeMatch(), this, InRun),
        getterOfProductsEndRun_(TypeMatch(), this, InRun),
        getterOfProductsBeginLumi_(edm::TypeMatch(), this, InLumi),
        getterOfProductsEndLumi_(edm::TypeMatch(), this, InLumi),
        getterOfProductsEvent_(edm::TypeMatch(), this) {
    callWhenNewProductsRegistered([this](edm::ProductDescription const& bd) {
      getterOfProductsBeginRun_(bd);
      getterOfProductsEndRun_(bd);
      getterOfProductsBeginLumi_(bd);
      getterOfProductsEndLumi_(bd);
      getterOfProductsEvent_(bd);
    });
  }

  void TestOutputWithGetterOfProducts::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    desc.addUntracked<unsigned int>("expectedSum", 0);
    descriptions.addDefault(desc);
  }

  void TestOutputWithGetterOfProducts::write(EventForOutput const& event) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEvent_.fillHandles(event, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProducts::writeLuminosityBlock(LuminosityBlockForOutput const& lumi) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEndLumi_.fillHandles(lumi, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProducts::writeRun(RunForOutput const& run) {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEndRun_.fillHandles(run, handles);
    sum_ += sumThings(handles);
  }

  std::shared_ptr<int> TestOutputWithGetterOfProducts::globalBeginRun(RunForOutput const& run) const {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsBeginRun_.fillHandles(run, handles);
    return std::make_shared<int>(sumThings(handles));
  }

  void TestOutputWithGetterOfProducts::globalEndRun(RunForOutput const& run) {
    sum_ += *runCache(run.index());
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEndRun_.fillHandles(run, handles);
    sum_ += sumThings(handles);
  }

  std::shared_ptr<int> TestOutputWithGetterOfProducts::globalBeginLuminosityBlock(
      LuminosityBlockForOutput const& lumi) const {
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsBeginLumi_.fillHandles(lumi, handles);
    return std::make_shared<int>(sumThings(handles));
  }

  void TestOutputWithGetterOfProducts::globalEndLuminosityBlock(LuminosityBlockForOutput const& lumi) {
    sum_ += *luminosityBlockCache(lumi.index());
    std::vector<Handle<edmtest::ThingCollection>> handles;
    getterOfProductsEndLumi_.fillHandles(lumi, handles);
    sum_ += sumThings(handles);
  }

  void TestOutputWithGetterOfProducts::endJob() {
    if (expectedSum_ != 0 && expectedSum_ != sum_) {
      throw cms::Exception("TestFailure") << "TestOutputWithGetterOfProducts::endJob, sum = " << sum_
                                          << " which is not equal to the expected value " << expectedSum_;
    }
  }

  int TestOutputWithGetterOfProducts::sumThings(std::vector<Handle<edmtest::ThingCollection>> const& collections) const {
    int sum = 0;
    for (auto const& collection : collections) {
      for (auto const& thing : *collection) {
        sum += thing.a;
      }
    }
    return sum;
  }
}  // namespace edm

using edm::TestOutputWithGetterOfProducts;
DEFINE_FWK_MODULE(TestOutputWithGetterOfProducts);
