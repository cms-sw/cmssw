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

  fractUpdateKey = ps.getUntrackedParameter<unsigned int>("FractUpdateKey", 1);
  fractUpdateEvF = ps.getUntrackedParameter<unsigned int>("FractUpdateEventFreq", 1);

  edm::ParameterSet psEffPar = ps.getUntrackedParameter<edm::ParameterSet>("effParameters");
  config.EFF_COLD_SIGFAIL = psEffPar.getUntrackedParameter<double>("sigfail_cold"  , 5.0);
  config.EFF_COLD_THRESHOLD = psEffPar.getUntrackedParameter<double>("threshold_cold", 0.1);
  config.EFF_ERR_SIGFAIL = psEffPar.getUntrackedParameter<double>("sigfail_err", 5.0);
  config.EFF_NODATA_THRESHOLD = psEffPar.getUntrackedParameter<double>("threshold_err", 0.1);
  config.EFF_HOT_SIGFAIL = psEffPar.getUntrackedParameter<double>("sigfail_hot"   , 2.0);
  config.EFF_HOT_THRESHOLD = psEffPar.getUntrackedParameter<double>("threshold_hot" , 0.1);
  config.EFF_NODATA_SIGFAIL = psEffPar.getUntrackedParameter<double>("sigfail_nodata", 5.0);
  config.EFF_NODATA_THRESHOLD = psEffPar.getUntrackedParameter<double>("threshold_nodata", 1.0);

  edm::FileInPath bookFile = ps.getParameter<edm::FileInPath>(PARAM_BOOKING_FILE);
  config.BOOKING_XML_FILE = bookFile.fullPath();
   
  config.provider = const_cast<CSCMonitorModuleCmn*>(this);

  dispatcher = new cscdqm::Dispatcher(&config);
  dbe = edm::Service<DQMStore>().operator->();

  // Prebook top level histograms
  dbe->setCurrentFolder(DIR_SUMMARY);
  dispatcher->getCollection()->book("EMU");
  bookedHisto.insert("EMU");

  // Booking parameters
  dbe->setCurrentFolder(DIR_SUMMARY_CONTENTS);
  bookFloat("CSC_SidePlus_Station01_Ring01", -1.0);
  bookFloat("CSC_SidePlus_Station01_Ring02", -1.0);
  bookFloat("CSC_SidePlus_Station01_Ring03", -1.0);
  bookFloat("CSC_SidePlus_Station01", -1.0);
  bookFloat("CSC_SidePlus_Station02_Ring01", -1.0);
  bookFloat("CSC_SidePlus_Station02_Ring02", -1.0);
  bookFloat("CSC_SidePlus_Station02", -1.0);
  bookFloat("CSC_SidePlus_Station03_Ring01", -1.0);
  bookFloat("CSC_SidePlus_Station03_Ring02", -1.0);
  bookFloat("CSC_SidePlus_Station03", -1.0);
  bookFloat("CSC_SidePlus_Station04", -1.0);
  bookFloat("CSC_SidePlus", -1.0);
  bookFloat("CSC_SideMinus_Station01_Ring01", -1.0);
  bookFloat("CSC_SideMinus_Station01_Ring02", -1.0);
  bookFloat("CSC_SideMinus_Station01_Ring03", -1.0);
  bookFloat("CSC_SideMinus_Station01", -1.0);
  bookFloat("CSC_SideMinus_Station02_Ring01", -1.0);
  bookFloat("CSC_SideMinus_Station02_Ring02", -1.0);
  bookFloat("CSC_SideMinus_Station02", -1.0);
  bookFloat("CSC_SideMinus_Station03_Ring01", -1.0);
  bookFloat("CSC_SideMinus_Station03_Ring02", -1.0);
  bookFloat("CSC_SideMinus_Station03", -1.0);
  bookFloat("CSC_SideMinus_Station04", -1.0);
  bookFloat("CSC_SideMinus", -1.0);

  dbe->setCurrentFolder(DIR_EVENTINFO);
  bookFloat(cscdqm::h::PAR_REPORT_SUMMARY, -1.0);

  //dispatcher->getCollection()->printCollection();
  //throw cscdqm::Exception("End of game");

}

CSCMonitorModuleCmn::~CSCMonitorModuleCmn() {
  delete dispatcher;
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
    
  dispatcher->processEvent(e, inputTag);

  // Update fractional histograms if appropriate
  if (config.getNEventsCSC() > 0 && fractUpdateKey.test(2) && (config.getNEvents() % fractUpdateEvF) == 0) {
    dispatcher->updateFractionHistos();
    dispatcher->updateEfficiencyHistos();
  }
    
}

void CSCMonitorModuleCmn::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
}

void CSCMonitorModuleCmn::endRun(const edm::Run& r, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::endJob() {
}


