#include <stdexcept>
#include <string>
#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>
#include <cstdlib>
#include <sstream>

typedef std::vector<int> Payload;

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
class OneIntRcd : public edm::eventsetup::EventSetupRecordImplementation<OneIntRcd> {};

#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
EVENTSETUP_RECORD_REG(OneIntRcd);



#include "FWCore/Framework/interface/MakerMacros.h"

namespace condtest {

  class TestUpdater : public edm::EDAnalyzer {
  public:
    explicit TestUpdater(edm::ParameterSet const&);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void beginRun(const edm::Run&, const edm::EventSetup&);
    virtual void endRun(const edm::Run& r, const edm::EventSetup&){}

    static void update(int run);
    
    int evCount;

  };


  TestUpdater::TestUpdater(edm::ParameterSet const&) : evCount(0){}

  void TestUpdater::beginRun(const edm::Run&, const edm::EventSetup&) {
    evCount=0;
  }

  void TestUpdater::analyze(const edm::Event& e, const edm::EventSetup& c) {
    ++evCount;

    if (0==e.id().run()%2 && evCount==3) update(e.id().run()+1);
    

    edm::ESHandle<std::vector<int> > h;
    c.get<OneIntRcd>().get(h);
    size_t number = (*h.product()).front();
    if (1==e.id().run()%2 && number!=e.id().run()) std::cout << "it was not updated!" << std::endl;
  }

  void TestUpdater::update(int run) {
    std::ostringstream ss;
    ss << "touch cfg.py; rm cfg.py; sed 's?_CurrentRun_?" << run << "?g' writeInt_cfg.py > cfg.py; cmsRun cfg.py";
    std::cout<< ss.str() << std::endl;
    // write run in db
    ::system(ss.str().c_str());

  }


  DEFINE_FWK_MODULE(TestUpdater);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(std::vector<int>);

#include "CondCore/PluginSystem/interface/registration_macros.h"

REGISTER_PLUGIN(OneIntRcd, std::vector<int> );
