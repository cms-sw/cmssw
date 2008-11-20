/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn.cc
 *
 *    Description:  Common CSC Monitor Module
 *
 *        Version:  1.0
 *        Created:  11/13/2008 02:31:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModuleCmn.h"

CSCMonitorModuleCmn::CSCMonitorModuleCmn(const edm::ParameterSet& ps) : inputTag(INPUT_TAG_LABEL) {

  CSCMonitorModuleCmn* hp = const_cast<CSCMonitorModuleCmn*>(this);
  collection = new cscdqm::Collection(hp);
  processor = new cscdqm::EventProcessor(hp);
  dbe = edm::Service<DQMStore>().operator->();

  fractUpdateKey = ps.getUntrackedParameter<unsigned int>("FractUpdateKey", 1);
  fractUpdateEvF = ps.getUntrackedParameter<unsigned int>("FractUpdateEventFreq", 1);

  edm::FileInPath bookFile = ps.getParameter<edm::FileInPath>(PARAM_BOOKING_FILE);
  collection->load(bookFile.fullPath());
   
  // Prebook top level histograms
  dbe->setCurrentFolder(DIR_EVENTINFO);
  collection->book("EventInfo");

  dbe->setCurrentFolder(DIR_SUMMARY);
  collection->book("EMU");

  //collection->printCollection();

}

CSCMonitorModuleCmn::~CSCMonitorModuleCmn() {
  delete collection;
  delete processor;
  while (!moCache.empty()) {
    delete moCache.begin()->second;
    moCache.erase(moCache.begin());
  }
}

void CSCMonitorModuleCmn::beginJob(const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::beginRun(const edm::Run& r, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::setup() {
}

void CSCMonitorModuleCmn::analyze(const edm::Event& e, const edm::EventSetup& c) {

  // Get crate mapping from database
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate);
  pcrate = hcrate.product();
    
  processor->processEvent(e, inputTag);

  LOG_INFO << "Should I update Fracts? nCSCEvents = " << processor->getNCSCEvents() << ", fractUpdateKey = " << fractUpdateKey << ", nEvents = " << processor->getNEvents() << ", fractUpdateEvF = " << fractUpdateEvF;
  
  // Update fractional histograms if appropriate
  if (processor->getNCSCEvents() > 0 && fractUpdateKey.test(2) && (processor->getNEvents() % fractUpdateEvF) == 0) {
    LOG_INFO << "Updating FRACTIONAL HISTOGRAMS!!!";
    processor->updateFractionHistos();
  }
    
}

void CSCMonitorModuleCmn::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
}

void CSCMonitorModuleCmn::endRun(const edm::Run& r, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::endJob() {
}


