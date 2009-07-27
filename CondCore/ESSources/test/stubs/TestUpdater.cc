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


#include "FWCore/Framework/interface/MakerMacros.h"

namespace condtest {

  class TestUpdater : public edm::EDAnalyzer {
  public:
    explicit TestUpdater(edm::ParameterSet const&);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void beginRun(const edm::Run&, const edm::EventSetup&);
    virtual void endRun(const edm::Run& r, const edm::EventSetup&);

    static void update(int run);
    
    int evCount;

  };


  TestUpdater::TestUpdater(edm::ParameterSet const&) : evCount(0){}
  void TestUpdater::beginRun(const edm::Run&, const edm::EventSetup&) {
    evCount=0;
  }

  void TestUpdater::analyze(const edm::Event& e, const edm::EventSetup& c) {
    ++evCount;

    if (evCount==3) update(e.id().run()+1);
    

    edm::ESHandle<std::vector<int> > h;
    // c.get<OneIntRcd>().get(h);
    // int number = (*h.product()).front();
  }

  void TestUpdater::update(int run) {
    std::ostringstream ss;
    ss << run;
   
    // write run in db
    ::system("touch cfg.py; rm cfg.py; sed 's?CurrentRun?'"+run.str()+"?g' writeInt_cfg.py > cfg.py; cmsRun cfg.py");

  }


  DEFINE_FWK_MODULE(TestUpdater);
}
