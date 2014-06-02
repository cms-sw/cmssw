#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>

namespace edm {

class DQMStoreAnalyzer: public EDAnalyzer {
   public:
   explicit DQMStoreAnalyzer(const ParameterSet&);
   ~DQMStoreAnalyzer();


   private:
virtual void analyze(Event const&, EventSetup const&);
virtual void beginJob();
virtual void endJob();

  virtual void beginRun(Run const&, EventSetup const&);
  virtual void endRun(Run const&, EventSetup const&);
  virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);
  virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&);

  void log(std::string x);
};

DQMStoreAnalyzer::DQMStoreAnalyzer(const edm::ParameterSet& iConfig) { log("Constructor"); }
DQMStoreAnalyzer::~DQMStoreAnalyzer() { log("Destructor"); }

void DQMStoreAnalyzer::log(std::string x) {
  std::cout << x << std::endl;
}

void DQMStoreAnalyzer::beginJob() { log("beginJob"); }
void DQMStoreAnalyzer::endJob() { log("endJob"); }

void DQMStoreAnalyzer::beginRun(Run const&, EventSetup const&) { log("beginRun"); }
void DQMStoreAnalyzer::endRun(Run const&, EventSetup const&) { log("endRun"); }

void DQMStoreAnalyzer::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) {
  log("beginLumi");

  edm::Service<DQMStore> store;
  std::vector<std::string> names = store->getSubdirs();
  for (std::vector<std::string>::const_iterator name = names.begin(); name != names.end(); ++name) {

    std::cout << "Directory: "<< *name << std::endl;
  }

  names = store->getMEs();
  for (std::vector<std::string>::const_iterator name = names.begin(); name != names.end(); ++name) {

    std::cout << "Name: "<< *name << std::endl;
  }
}


void DQMStoreAnalyzer::endLuminosityBlock(LuminosityBlock const&, EventSetup const&) { log("endLuminosityBlock"); }

void DQMStoreAnalyzer::analyze(Event const&, EventSetup const&) { log("analyze"); }


} /* namespace */


using edm::DQMStoreAnalyzer;
DEFINE_FWK_MODULE(DQMStoreAnalyzer);
