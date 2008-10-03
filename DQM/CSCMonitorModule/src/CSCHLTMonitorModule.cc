/*
 * =====================================================================================
 *
 *       Filename:  CSCHLTMonitorModule.cc
 *
 *    Description:  CSC HLT Monitor module class
 *
 *        Version:  1.0
 *        Created:  09/15/2008 01:26:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCHLTMonitorModule.h"
#include "CSCUtilities.cc"

/**
 * @brief  MonitorModule Constructor
 * @param  ps ParameterSet
 * @return
 */
CSCHLTMonitorModule::CSCHLTMonitorModule(const edm::ParameterSet& ps){

  parameters = ps;

  examinerMask   = parameters.getUntrackedParameter<unsigned int>("ExaminerMask", 0x7FB7BF6);
  examinerForce  = parameters.getUntrackedParameter<bool>("ExaminerForce", false);
  examinerOutput = parameters.getUntrackedParameter<bool>("ExaminerOutput", false);
  examinerCRCKey = parameters.getUntrackedParameter<unsigned int>("ExaminerCRCKey", 0);
  inputObjectsTag = parameters.getUntrackedParameter<edm::InputTag>("InputObjects", (edm::InputTag)"source");
  monitorName = parameters.getUntrackedParameter<std::string>("monitorName", "CSC");

  std::vector<unsigned int> vIds = parameters.getUntrackedParameter<std::vector<unsigned int> >("FEDIds");
  for(std::vector<unsigned int>::iterator i = vIds.begin(); i != vIds.end(); i++) {
    fedIds.insert(*i);
  }

  rootDir = monitorName + "/";
  nEvents = 0;

  // Get back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  this->init = false;

}

/**
 * @brief  MonitorModule Destructor
 * @param
 * @return
 */
CSCHLTMonitorModule::~CSCHLTMonitorModule(){

}

/**
 * @brief  Function that is being executed prior job. Actuall histogram
 * bookings are done here. All initialization tasks as well.
 * @param  c Event setup object
 * @return
 */
void CSCHLTMonitorModule::beginJob(const edm::EventSetup& c){

}

void CSCHLTMonitorModule::setup() {

  // Base folder for the contents of this job
  dbe->setCurrentFolder(rootDir + FED_FOLDER);

  unsigned int fsize = fedIds.size();
  mes.insert(std::make_pair("FEDEntries", dbe->book1D("FEDEntries", "CSC FED Entries", fsize, 0, fsize)));
  mes.insert(std::make_pair("FEDFatal", dbe->book1D("FEDFatal", "CSC FED Fatal Errors", fsize, 0, fsize)));
  mes.insert(std::make_pair("FEDNonFatal", dbe->book1D("FEDNonFatal", "CSC FED Non Fatal Errors", fsize, 0, fsize)));

  for(MeMap::iterator iter = mes.begin(); iter != mes.end(); iter++) {
    MonitorElement *me = iter->second; 
    TH1 *h = me->getTH1();
    me->setAxisTitle("FED Id", 1);
    me->setAxisTitle("# of Events", 2);
    h->SetOption("bar1text");
    h->SetStats(0);
    unsigned int index = 1;
    for (std::set<unsigned int>::iterator i = fedIds.begin(); i != fedIds.end(); i++ ) {
      std::stringstream out;
      out << *i;
      h->GetXaxis()->SetBinLabel(index, out.str().c_str());
      index++;
    }
  }
  
  this->init = true;

}

/**
 * @brief  Main Analyzer function that receives Events andi starts
 * the acctual analysis (histogram filling) and so on chain.
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCHLTMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  // Get crate mapping from database
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate);

  // Lets initialize MEs if it was not done so before 
  if (!this->init) {
    this->setup();
  }

  // Pass event to monitoring chain
  monitorEvent(e);

}


/**
 * @brief  Get FED index on FED id
 * @param  fedId FED ID
 * @param  index FED index in histograms
 * @return true if FED was found, false - otherwise
 */
const bool CSCHLTMonitorModule::fedIndex(const unsigned int fedId, unsigned int& index) const {
  index = 0;
  for (std::set<unsigned int>::iterator i = fedIds.begin(); i != fedIds.end(); i++ ) {
    if (*i == fedId) {
      //LOGINFO("FED identified") << "FED id: " << fedId << " has index: " << index;
      return true;
    }
    index++;
  }
  LOGINFO("FED id not defined") << "FED id: " << fedId << " is not defined in cfg file.";
  return false;
}

/**
 * @brief  Function that is being executed at the very end of the job.
 * Histogram savings, calculation of fractional histograms, etc. should be done
 * here.
 * @param
 * @return
 */
void CSCHLTMonitorModule::endJob(void){

}

void CSCHLTMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}

void CSCHLTMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context) {

}

void CSCHLTMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}

