
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyServiceE0.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/Utilities/interface/Exception.h"

//NOTE: I need to open a 'back door' so I can test ServiceManager 'inheritance'
#define private public
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#undef private

#include <cstdlib>
#include <vector>
#include <memory>
#include <iostream>

int main() try {
  using namespace edm::serviceregistry;

  // We must initialize the plug-in manager first
  edm::AssertHandler ah;

  // These services check the order their constructor, postBeginJob,
  // postEndJob, and destructor are called and if they are not
  // called in the correct order they will abort
  typedef testserviceregistry::DummyServiceE0 Service0;
  typedef testserviceregistry::DummyServiceA1 Service1;
  typedef testserviceregistry::DummyServiceD2 Service2;
  typedef testserviceregistry::DummyServiceB3 Service3;
  typedef testserviceregistry::DummyServiceC4 Service4;

  // Build the services in a manner similar to the way the are constructed
  // in a cmsRun job.  Build one service directly, then three based
  // on parameter sets, then another one directly.  ServiceB3
  // includes an explicit dependence on ServiceD2 so build on
  // demand is also tested.

  std::vector<edm::ParameterSet> vps;
  auto legacy = std::make_shared<ServicesManager>(vps);

  edm::ActivityRegistry ar;
  edm::ParameterSet pset;
  std::auto_ptr<Service0> s0(new Service0(pset, ar));  
  auto wrapper = std::make_shared<ServiceWrapper<Service0> >(s0);
  legacy->put(wrapper);
  legacy->copySlotsFrom(ar);
  edm::ServiceToken legacyToken(legacy);

  std::vector<edm::ParameterSet> vps1;

  edm::ParameterSet ps1;
  std::string typeName1("DummyServiceA1");
  ps1.addParameter("@service_type", typeName1);
  vps1.push_back(ps1);
      
  // The next two are intentionally swapped to test build
  // on demand feature.  DummyServiceB3 depends on DummyServiceD2
  // so they should end up getting built in the reverse of the
  // order specified here.

  edm::ParameterSet ps3;
  std::string typeName3("DummyServiceB3");
  ps3.addParameter("@service_type", typeName3);
  vps1.push_back(ps3);

  edm::ParameterSet ps2;
  std::string typeName2("DummyServiceD2");
  ps2.addParameter("@service_type", typeName2);
  vps1.push_back(ps2);

  auto legacy2 = std::make_shared<ServicesManager>(legacyToken, kTokenOverrides, vps1);
  edm::ServiceToken legacyToken2(legacy2);


  ServicesManager sm(legacyToken2, kOverlapIsError, vps);

  edm::ActivityRegistry ar4;
  edm::ParameterSet pset4;
  std::auto_ptr<Service4> s4(new Service4(pset4, ar4));  
  auto wrapper4 = std::make_shared<ServiceWrapper<Service4> >(s4);
  sm.put(wrapper4);
  sm.copySlotsFrom(ar4);


  edm::ActivityRegistry actReg;
  sm.connectTo(actReg);
  actReg.postBeginJobSignal_();
  actReg.postEndJobSignal_();

  return 0;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return 1;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
