/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/ProductResolversFactory.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

#include "catch2/catch_all.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "makeDummyProcessConfiguration.h"

namespace {
  std::map<std::string, std::shared_ptr<edm::ProductDescription>> productDescriptions_;
  std::map<std::string, std::shared_ptr<edm::ProcessConfiguration>> processConfigurations_;
  std::shared_ptr<edm::SignallingProductRegistryFiller> pProductRegistry_;
  std::shared_ptr<edm::LuminosityBlockPrincipal> lbp_;
  std::shared_ptr<edm::EventPrincipal> pEvent_;
  edm::EventID eventID_;
  edm::HistoryAppender historyAppender_;

  std::shared_ptr<edm::ProcessConfiguration> fake_single_module_process(
      std::string const& tag,
      std::string const& processName,
      edm::ParameterSet const& moduleParams,
      std::string const& release = edm::getReleaseVersion()) {
    edm::ParameterSet processParams;
    processParams.addParameter(processName, moduleParams);
    processParams.addParameter<std::string>("@process_name", processName);

    processParams.registerIt();
    auto result = edmtest::makeSharedDummyProcessConfiguration(processName, processParams.id());
    processConfigurations_[tag] = result;
    return result;
  }

  std::shared_ptr<edm::ProductDescription> fake_single_process_branch(
      std::string const& tag, std::string const& processName, std::string const& productInstanceName = std::string()) {
    std::string moduleLabel = processName + "dummyMod";
    std::string moduleClass("DummyModule");
    edm::TypeID dummyType(typeid(edmtest::DummyProduct));
    edm::ParameterSet modParams;
    modParams.addParameter<std::string>("@module_type", moduleClass);
    modParams.addParameter<std::string>("@module_label", moduleLabel);
    modParams.registerIt();
    std::shared_ptr<edm::ProcessConfiguration> process(fake_single_module_process(tag, processName, modParams));

    auto result = std::make_shared<edm::ProductDescription>(
        edm::InEvent, moduleLabel, processName, productInstanceName, dummyType);
    productDescriptions_[tag] = result;
    return result;
  }

  void setUp() {
    // Making a functional EventPrincipal is not trivial, so we do it
    // all here.
    eventID_ = edm::EventID(101, 1, 20);

    // We can only insert products registered in the ProductRegistry.
    pProductRegistry_.reset(new edm::SignallingProductRegistryFiller);
    pProductRegistry_->addProduct(*fake_single_process_branch("hlt", "HLT"));
    pProductRegistry_->addProduct(*fake_single_process_branch("prod", "PROD"));
    pProductRegistry_->addProduct(*fake_single_process_branch("test", "TEST"));
    pProductRegistry_->addProduct(*fake_single_process_branch("user", "USER"));
    pProductRegistry_->addProduct(*fake_single_process_branch("rick", "USER2", "rick"));
    pProductRegistry_->setProcessOrder({"USER2", "USER", "TEST", "PROD", "HLT"});
    pProductRegistry_->setFrozen();
    auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
    branchIDListHelper->updateFromRegistry(pProductRegistry_->registry());

    // Put products we'll look for into the EventPrincipal.
    {
      typedef edmtest::DummyProduct PRODUCT_TYPE;
      typedef edm::Wrapper<PRODUCT_TYPE> WDP;

      std::unique_ptr<edm::WrapperBase> product = std::make_unique<WDP>(std::make_unique<PRODUCT_TYPE>());

      std::string tag("rick");
      assert(productDescriptions_[tag]);
      edm::ProductDescription branch = *productDescriptions_[tag];

      branch.init();

      edm::ProductRegistry::ProductList const& pl = pProductRegistry_->registry().productList();
      edm::BranchKey const bk(branch);
      edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);

      edm::ProductDescription const branchFromRegistry(it->second);

      std::vector<edm::BranchID> const ids;
      edm::ProductProvenance prov(branchFromRegistry.branchID(), ids);

      std::shared_ptr<edm::ProcessConfiguration> process(processConfigurations_[tag]);
      assert(process);
      std::string uuid = edm::createGlobalIdentifier();
      edm::Timestamp now(1234567UL);
      auto pRegistry = std::make_shared<edm::ProductRegistry const>(pProductRegistry_->registry());
      auto rp = std::make_shared<edm::RunPrincipal>(
          pRegistry, edm::productResolversFactory::makePrimary, *process, &historyAppender_, 0);
      rp->setAux(edm::RunAuxiliary(eventID_.run(), now, now));
      edm::LuminosityBlockAuxiliary lumiAux(rp->run(), 1, now, now);
      lbp_ = std::make_shared<edm::LuminosityBlockPrincipal>(
          pRegistry, edm::productResolversFactory::makePrimary, *process, &historyAppender_, 0);
      lbp_->setAux(lumiAux);
      lbp_->setRunPrincipal(rp);
      edm::EventAuxiliary eventAux(eventID_, uuid, now, true);
      pEvent_.reset(new edm::EventPrincipal(pRegistry,
                                            edm::productResolversFactory::makePrimary,
                                            branchIDListHelper,
                                            *process,
                                            &historyAppender_,
                                            edm::StreamID::invalidStreamID()));
      pEvent_->fillEventPrincipal(eventAux, nullptr);
      pEvent_->setLuminosityBlockPrincipal(lbp_.get());
      pEvent_->put(branchFromRegistry, std::move(product), prov);
    }
    REQUIRE(pEvent_->size() == 1);
  }

}  // namespace

TEST_CASE("EventPrincipal", "[Framework]") {
  setUp();

  SECTION("failgetbyIdTest") {
    edm::ProductID invalid;
    REQUIRE_THROWS_AS(pEvent_->getByProductID(invalid), edm::Exception);

    edm::ProductID notpresent(0, 10000);
    edm::BasicHandle h(pEvent_->getByProductID(notpresent));
    REQUIRE(h.failedToGet());
  }

  SECTION("failgetbyLabelTest") {
    edmtest::IntProduct dummy;
    edm::TypeID tid(dummy);

    std::string label("this does not exist");

    edm::BasicHandle h(
        pEvent_->getByLabel(edm::PRODUCT_TYPE, tid, label, std::string(), std::string(), nullptr, nullptr, nullptr));
    REQUIRE(h.failedToGet());
  }

  SECTION("failgetbyInvalidIdTest") {
    //put_a_dummy_product("HLT");
    //put_a_product<edmtest::DummyProduct>(pProdConfig_, label);

    edm::ProductID id;
    REQUIRE_THROWS_AS(pEvent_->getByProductID(id), edm::Exception);
  }

  SECTION("failgetProvenanceTest") {
    edm::BranchID id;
    REQUIRE_THROWS_AS(pEvent_->getProvenance(id), edm::Exception);
  }
}
