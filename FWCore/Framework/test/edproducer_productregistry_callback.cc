/**
   \file
   test for ProductRegistry 

   \author Stefano ARGIRO
   \date 21 July 2005
*/

#include <iostream>
#include "cppunit/extensions/HelperMacros.h"
#include <memory>
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "FWCore/Framework/interface/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Framework/interface/maker/WorkerMaker.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/TypeID.h"

class testEDProducerProductRegistryCallback : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEDProducerProductRegistryCallback);

  CPPUNIT_TEST_EXCEPTION(testCircularRef, cms::Exception);
  CPPUNIT_TEST_EXCEPTION(testCircularRef2, cms::Exception);
  CPPUNIT_TEST(testTwoListeners);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void testCircularRef();
  void testCircularRef2();
  void testTwoListeners();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEDProducerProductRegistryCallback);

using namespace edm;

namespace {
  class TestMod : public global::EDProducer<> {
  public:
    explicit TestMod(ParameterSet const& p);

    void produce(StreamID, Event& e, EventSetup const&) const override;

    void listen(BranchDescription const&);
  };

  TestMod::TestMod(ParameterSet const&) { produces<int>(); }

  void TestMod::produce(StreamID, Event&, EventSetup const&) const {}

  class ListenMod : public global::EDProducer<> {
  public:
    explicit ListenMod(ParameterSet const&);
    void produce(StreamID, Event& e, EventSetup const&) const override;
    void listen(BranchDescription const&);
  };

  ListenMod::ListenMod(ParameterSet const&) {
    callWhenNewProductsRegistered(
        [this](BranchDescription const& branchDescription) { this->listen(branchDescription); });
  }
  void ListenMod::produce(StreamID, Event&, EventSetup const&) const {}

  void ListenMod::listen(BranchDescription const& iDesc) {
    edm::TypeID intType(typeid(int));
    //std::cout << "see class " << iDesc.typeName() << std::endl;
    if (iDesc.friendlyClassName() == intType.friendlyClassName()) {
      produces<int>(iDesc.moduleLabel() + "-" + iDesc.productInstanceName());
      //std::cout << iDesc.moduleLabel() << "-" << iDesc.productInstanceName() << std::endl;
    }
  }

  class ListenFloatMod : public global::EDProducer<> {
  public:
    explicit ListenFloatMod(ParameterSet const&);
    void produce(StreamID, Event& e, EventSetup const&) const;
    void listen(BranchDescription const&);
  };

  ListenFloatMod::ListenFloatMod(ParameterSet const&) {
    callWhenNewProductsRegistered(
        [this](BranchDescription const& branchDescription) { this->listen(branchDescription); });
  }
  void ListenFloatMod::produce(StreamID, Event&, EventSetup const&) const {}

  void ListenFloatMod::listen(BranchDescription const& iDesc) {
    edm::TypeID intType(typeid(int));
    //std::cout <<"see class "<<iDesc.typeName()<<std::endl;
    if (iDesc.friendlyClassName() == intType.friendlyClassName()) {
      produces<float>(iDesc.moduleLabel() + "-" + iDesc.productInstanceName());
      //std::cout <<iDesc.moduleLabel()<<"-"<<iDesc.productInstanceName()<<std::endl;
    }
  }
}  // namespace

void testEDProducerProductRegistryCallback::testCircularRef() {
  using namespace edm;

  SignallingProductRegistry preg;

  //Need access to the ConstProductRegistry service
  auto cReg = std::make_unique<ConstProductRegistry>(preg);
  ServiceToken token = ServiceRegistry::createContaining(std::move(cReg));
  ServiceRegistry::Operate startServices(token);

  std::unique_ptr<Maker> f = std::make_unique<WorkerMaker<TestMod>>();

  ParameterSet p1;
  p1.addParameter("@module_type", std::string("TestMod"));
  p1.addParameter("@module_label", std::string("t1"));
  p1.addParameter("@module_edm_type", std::string("EDProducer"));
  p1.registerIt();

  ParameterSet p2;
  p2.addParameter("@module_type", std::string("TestMod"));
  p2.addParameter("@module_label", std::string("t2"));
  p2.addParameter("@module_edm_type", std::string("EDProducer"));
  p2.registerIt();

  edm::ExceptionToActionTable table;
  edm::PreallocationConfiguration prealloc;

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto pc =
      std::make_shared<ProcessConfiguration>("PROD", dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID());

  edm::MakeModuleParams params1(&p1, preg, &prealloc, pc);
  edm::MakeModuleParams params2(&p2, preg, &prealloc, pc);

  std::unique_ptr<Maker> lM = std::make_unique<WorkerMaker<ListenMod>>();
  ParameterSet l1;
  l1.addParameter("@module_type", std::string("ListenMod"));
  l1.addParameter("@module_label", std::string("l1"));
  l1.addParameter("@module_edm_type", std::string("EDProducer"));
  l1.registerIt();

  ParameterSet l2;
  l2.addParameter("@module_type", std::string("ListenMod"));
  l2.addParameter("@module_label", std::string("l2"));
  l2.addParameter("@module_edm_type", std::string("EDProducer"));
  l2.registerIt();

  edm::MakeModuleParams paramsl1(&l1, preg, &prealloc, pc);
  edm::MakeModuleParams paramsl2(&l2, preg, &prealloc, pc);

  signalslot::Signal<void(const ModuleDescription&)> aSignal;

  auto m1 = f->makeModule(params1, aSignal, aSignal);
  std::unique_ptr<Worker> w1 = m1->makeWorker(&table);
  auto ml1 = lM->makeModule(paramsl1, aSignal, aSignal);
  std::unique_ptr<Worker> wl1 = ml1->makeWorker(&table);
  auto ml2 = lM->makeModule(paramsl2, aSignal, aSignal);
  std::unique_ptr<Worker> wl2 = ml2->makeWorker(&table);
  auto m2 = f->makeModule(params2, aSignal, aSignal);
  std::unique_ptr<Worker> w2 = m2->makeWorker(&table);

  //Should be 5 products
  // 1 from the module 't1'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1'
  //    1 from 'l2' in response to 't1'
  //       1 from 'l1' in response to 'l2'
  // 1 from the module 't2'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1'
  //    1 from 'l2' in response to 't2'
  //       1 from 'l1' in response to 'l2'
  //std::cout <<"# products "<<preg.size()<<std::endl;
  CPPUNIT_ASSERT(10 == preg.size());
}

void testEDProducerProductRegistryCallback::testCircularRef2() {
  using namespace edm;

  SignallingProductRegistry preg;

  //Need access to the ConstProductRegistry service
  auto cReg = std::make_unique<ConstProductRegistry>(preg);
  ServiceToken token = ServiceRegistry::createContaining(std::move(cReg));
  ServiceRegistry::Operate startServices(token);

  std::unique_ptr<Maker> f = std::make_unique<WorkerMaker<TestMod>>();

  ParameterSet p1;
  p1.addParameter("@module_type", std::string("TestMod"));
  p1.addParameter("@module_label", std::string("t1"));
  p1.addParameter("@module_edm_type", std::string("EDProducer"));
  p1.registerIt();

  ParameterSet p2;
  p2.addParameter("@module_type", std::string("TestMod"));
  p2.addParameter("@module_label", std::string("t2"));
  p2.addParameter("@module_edm_type", std::string("EDProducer"));
  p2.registerIt();

  edm::ExceptionToActionTable table;
  edm::PreallocationConfiguration prealloc;

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto pc =
      std::make_shared<ProcessConfiguration>("PROD", dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID());

  edm::MakeModuleParams params1(&p1, preg, &prealloc, pc);
  edm::MakeModuleParams params2(&p2, preg, &prealloc, pc);

  std::unique_ptr<Maker> lM = std::make_unique<WorkerMaker<ListenMod>>();
  ParameterSet l1;
  l1.addParameter("@module_type", std::string("ListenMod"));
  l1.addParameter("@module_label", std::string("l1"));
  l1.addParameter("@module_edm_type", std::string("EDProducer"));
  l1.registerIt();

  ParameterSet l2;
  l2.addParameter("@module_type", std::string("ListenMod"));
  l2.addParameter("@module_label", std::string("l2"));
  l2.addParameter("@module_edm_type", std::string("EDProducer"));
  l2.registerIt();

  edm::MakeModuleParams paramsl1(&l1, preg, &prealloc, pc);
  edm::MakeModuleParams paramsl2(&l2, preg, &prealloc, pc);

  signalslot::Signal<void(const ModuleDescription&)> aSignal;
  auto ml1 = lM->makeModule(paramsl1, aSignal, aSignal);
  std::unique_ptr<Worker> wl1 = ml1->makeWorker(&table);
  auto ml2 = lM->makeModule(paramsl2, aSignal, aSignal);
  std::unique_ptr<Worker> wl2 = ml2->makeWorker(&table);
  auto m1 = f->makeModule(params1, aSignal, aSignal);
  std::unique_ptr<Worker> w1 = m1->makeWorker(&table);
  auto m2 = f->makeModule(params2, aSignal, aSignal);
  std::unique_ptr<Worker> w2 = m2->makeWorker(&table);

  //Would be 10 products
  // 1 from the module 't1'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1' <-- circular
  //    1 from 'l2' in response to 't1'                  |
  //       1 from 'l1' in response to 'l2' <-- circular /
  // 1 from the module 't2'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1'
  //    1 from 'l2' in response to 't2'
  //       1 from 'l1' in response to 'l2'
  //std::cout <<"# products "<<preg.size()<<std::endl;
  CPPUNIT_ASSERT(10 == preg.size());
}

void testEDProducerProductRegistryCallback::testTwoListeners() {
  using namespace edm;

  SignallingProductRegistry preg;

  //Need access to the ConstProductRegistry service
  auto cReg = std::make_unique<ConstProductRegistry>(preg);
  ServiceToken token = ServiceRegistry::createContaining(std::move(cReg));
  ServiceRegistry::Operate startServices(token);

  std::unique_ptr<Maker> f = std::make_unique<WorkerMaker<TestMod>>();

  ParameterSet p1;
  p1.addParameter("@module_type", std::string("TestMod"));
  p1.addParameter("@module_label", std::string("t1"));
  p1.addParameter("@module_edm_type", std::string("EDProducer"));
  p1.registerIt();

  ParameterSet p2;
  p2.addParameter("@module_type", std::string("TestMod"));
  p2.addParameter("@module_label", std::string("t2"));
  p2.addParameter("@module_edm_type", std::string("EDProducer"));
  p2.registerIt();

  edm::ExceptionToActionTable table;
  edm::PreallocationConfiguration prealloc;

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto pc =
      std::make_shared<ProcessConfiguration>("PROD", dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID());

  edm::MakeModuleParams params1(&p1, preg, &prealloc, pc);
  edm::MakeModuleParams params2(&p2, preg, &prealloc, pc);

  std::unique_ptr<Maker> lM = std::make_unique<WorkerMaker<ListenMod>>();
  ParameterSet l1;
  l1.addParameter("@module_type", std::string("ListenMod"));
  l1.addParameter("@module_label", std::string("l1"));
  l1.addParameter("@module_edm_type", std::string("EDProducer"));
  l1.registerIt();

  std::unique_ptr<Maker> lFM = std::make_unique<WorkerMaker<ListenFloatMod>>();
  ParameterSet l2;
  l2.addParameter("@module_type", std::string("ListenMod"));
  l2.addParameter("@module_label", std::string("l2"));
  l2.addParameter("@module_edm_type", std::string("EDProducer"));
  l2.registerIt();

  edm::MakeModuleParams paramsl1(&l1, preg, &prealloc, pc);
  edm::MakeModuleParams paramsl2(&l2, preg, &prealloc, pc);

  signalslot::Signal<void(const ModuleDescription&)> aSignal;
  auto m1 = f->makeModule(params1, aSignal, aSignal);
  std::unique_ptr<Worker> w1 = m1->makeWorker(&table);
  auto ml1 = lM->makeModule(paramsl1, aSignal, aSignal);
  std::unique_ptr<Worker> wl1 = ml1->makeWorker(&table);
  auto ml2 = lFM->makeModule(paramsl2, aSignal, aSignal);
  std::unique_ptr<Worker> wl2 = ml2->makeWorker(&table);
  auto m2 = f->makeModule(params2, aSignal, aSignal);
  std::unique_ptr<Worker> w2 = m2->makeWorker(&table);

  //Should be 8 products
  // 1 from the module 't1'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1'
  //    1 from 'l2' in response to 't1'
  // 1 from the module 't2'
  //    1 from 'l1' in response
  //       1 from 'l2' in response to 'l1'
  //    1 from 'l2' in response to 't2'
  //std::cout <<"# products "<<preg.size()<<std::endl;
  CPPUNIT_ASSERT(8 == preg.size());
}
