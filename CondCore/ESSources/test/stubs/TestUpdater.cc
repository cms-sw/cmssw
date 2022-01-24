#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <sstream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

typedef std::vector<int> Payload;

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
class OneIntRcd : public edm::eventsetup::EventSetupRecordImplementation<OneIntRcd> {};

#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
EVENTSETUP_RECORD_REG(OneIntRcd);

#include "FWCore/Framework/interface/MakerMacros.h"

namespace condtest {

  class TestUpdater : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    explicit TestUpdater(edm::ParameterSet const&);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void endRun(const edm::Run&, const edm::EventSetup&) override;

    static void update(int run);

    int evCount;

  private:
    const edm::ESGetToken<std::vector<int>, OneIntRcd> theIntToken_;
  };

  TestUpdater::TestUpdater(edm::ParameterSet const&) : evCount(0), theIntToken_(esConsumes()) {}

  void TestUpdater::beginRun(const edm::Run&, const edm::EventSetup&) { evCount = 0; }

  void TestUpdater::endRun(const edm::Run&, const edm::EventSetup&) {}

  void TestUpdater::analyze(const edm::Event& e, const edm::EventSetup& c) {
    ++evCount;

    if (0 == e.id().run() % 2 && evCount == 3)
      update(e.id().run() + 1);

    size_t number = (c.getData(theIntToken_)).front();
    if (1 == e.id().run() % 2 && number != e.id().run())
      edm::LogPrint("TestUpdater") << "it was not updated!";
  }

  void TestUpdater::update(int run) {
    std::ostringstream ss;
    ss << "touch cfg.py; rm cfg.py; sed 's?_CurrentRun_?" << run << "?g' writeInt_cfg.py > cfg.py; cmsRun cfg.py";
    edm::LogPrint("TestUpdater") << ss.str();
    // write run in db
    ::system(ss.str().c_str());
  }

  DEFINE_FWK_MODULE(TestUpdater);
}  // namespace condtest

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(std::vector<int>);

#include "CondCore/ESSources/interface/registration_macros.h"

REGISTER_PLUGIN(OneIntRcd, std::vector<int>);
