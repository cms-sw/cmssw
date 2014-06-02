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

class DQMStoreWriter: public EDAnalyzer {
   public:
   explicit DQMStoreWriter(const ParameterSet&);
   ~DQMStoreWriter();


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

DQMStoreWriter::DQMStoreWriter(const edm::ParameterSet& iConfig) { log("Constructor"); }
DQMStoreWriter::~DQMStoreWriter() { log("Destructor"); }

void DQMStoreWriter::log(std::string x) {
  std::cout << x << std::endl;
}

void DQMStoreWriter::beginJob() { log("beginJob"); }
void DQMStoreWriter::endJob() { log("endJob"); }

void DQMStoreWriter::beginRun(Run const&, EventSetup const&) { log("beginRun"); }
void DQMStoreWriter::endRun(Run const&, EventSetup const&) { log("endRun"); }

void DQMStoreWriter::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) {
  edm::Service<DQMStore> store;
  store->setCurrentFolder("tdir");

  MonitorElement *mi = store->bookInt("testInt");
  MonitorElement *mf = store->bookFloat("testFloat");
  MonitorElement *ms = store->bookString("testString", "15");

  MonitorElement *mh = store->book1D("testHisto", "test test", 100, 1, 100);
  MonitorElement *mg = store->book1D("testGlobal", "test test", 100, 1, 100);

  mi->Fill(15);
  mf->Fill(15);
  mh->Fill(15);
  mg->Fill(15);

  mg->setLumiFlag();
  ms->getLumiFlag();

  std::cout << "Wrote histo." << std::endl;
}


void DQMStoreWriter::endLuminosityBlock(LuminosityBlock const&, EventSetup const&) { log("endLuminosityBlock"); }

void DQMStoreWriter::analyze(Event const&, EventSetup const&) { log("analyze"); }


} /* namespace */

using edm::DQMStoreWriter;
DEFINE_FWK_MODULE(DQMStoreWriter);
