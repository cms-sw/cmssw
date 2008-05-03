/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule.cc
 *
 *    Description:  CSC Monitor module class
 *
 *        Version:  1.0
 *        Created:  04/18/2008 02:26:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include "csc_utilities.cc"

/**
 * @brief  MonitorModule Constructor
 * @param  ps ParameterSet
 * @return
 */
CSCMonitorModule::CSCMonitorModule(const edm::ParameterSet& ps){

  parameters=ps;
  getCSCTypeToBinMap(tmap);

  edm::FileInPath fp;

  std::vector<unsigned> ddus = parameters.getUntrackedParameter< std::vector<unsigned> >("bookDDU");
  if (ddus.size() > 1) {
    loadDDU = ddus[1];
    loadDDU <<= 18;
    BitsetDDU minusDDU(ddus[0]);
    loadDDU |= minusDDU;
  }
  
  examinerMask   = parameters.getUntrackedParameter<unsigned int>("ExaminerMask", 0x7FB7BF6);
  examinerForce  = parameters.getUntrackedParameter<bool>("ExaminerForce", false);
  examinerOutput = parameters.getUntrackedParameter<bool>("ExaminerOutput", false);
  examinerCRCKey = parameters.getUntrackedParameter<unsigned int>("ExaminerCRCKey", 0);

  // Initialize some variables
  inputObjectsTag = parameters.getUntrackedParameter<edm::InputTag>("InputObjects", (edm::InputTag)"source");
  monitorName = parameters.getUntrackedParameter<std::string>("monitorName", "CSC");
  fp = parameters.getParameter<edm::FileInPath>("BookingFile");
  bookingFile = fp.fullPath();

  rootDir = monitorName + "/";
  nEvents = 0;

  // Loading histogram collection from XML file
  if(loadCollection()) {
    LOGERROR("initialize") << "Histogram booking failed .. exiting.";
    return;
  }

  // Get back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  this->init = false;

}

/**
 * @brief  MonitorModule Destructor
 * @param
 * @return
 */
CSCMonitorModule::~CSCMonitorModule(){

}

/**
 * @brief  Function that is being executed prior job. Actuall histogram
 * bookings are done here. All initialization tasks as well.
 * @param  c Event setup object
 * @return
 */
void CSCMonitorModule::beginJob(const edm::EventSetup& c){

}

void CSCMonitorModule::setup() {

  // Base folder for the contents of this job
  dbe->setCurrentFolder(rootDir);

  // Book EMU level histograms
  book("EMU");

  // Book DDU histograms
  for (int d = 1; d <= 36; d++) {
    if(!loadDDU.test(d - 1)) continue;
    std::string buffer;
    dbe->setCurrentFolder(rootDir + getDDUTag(d, buffer));
    book("DDU");
  }
  LOGINFO("DDU histograms") << " # of DDU to be monitored = " << loadDDU.count() << " following bitset = " << loadDDU;

  this->init = true;

}

/**
 * @brief  Main Analyzer function that receives Events andi starts
 * the acctual analysis (histogram filling) and so on chain.
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  // Get crate mapping from database
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate);
  pcrate = hcrate.product();

  if (!this->init) this->setup();

  // Pass event to monitoring chain
  monitorEvent(e);

}

/**
 * @brief  Function that is being executed at the very end of the job.
 * Histogram savings, calculation of fractional histograms, etc. should be done
 * here.
 * @param
 * @return
 */
void CSCMonitorModule::endJob(void){

}

void CSCMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}

void CSCMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context) {
  updateFracHistos();
}

void CSCMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}

void CSCMonitorModule::getCSCFromMap(int crate, int slot, int& csctype, int& cscposition) {

  CSCDetId cid = pcrate->detId(crate, slot, 0, 0);
  cscposition  = cid.chamber();
  int iring    = cid.ring();
  int istation = cid.station();
  int iendcap  = cid.endcap();
  
  std::string tlabel = getCSCTypeLabel(iendcap, istation, iring);
  std::map<std::string,int>::const_iterator it = tmap.find(tlabel);
  if (it != tmap.end()) {
    csctype = it->second;
  } else {
    csctype = 0;
  }

}  
