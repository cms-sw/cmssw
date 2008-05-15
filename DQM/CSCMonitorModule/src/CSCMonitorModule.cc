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
  initialize();
  getCSCTypeToBinMap(tmap);

}

/**
 * @brief Initialize variables and the stuff
 * @param  
 * @return 
 */
void CSCMonitorModule::initialize() {

  edm::FileInPath fp;

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

  // Load CSC mapping object
  cscMapping  = CSCReadoutMappingFromFile(parameters); 

  // Booking histograms aka loading collection
  if(loadCollection()) {
    LOGERROR("initialize") << "Histogram booking failed .. exiting.";
    return;
  }

  // Get back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  // Base folder for the contents of this job
  dbe->setCurrentFolder(rootDir);

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

  // Get back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  // Base folder for the contents of this job
  dbe->setCurrentFolder(rootDir);

  book("EMU");
}

/**
 * @brief  Main Analyzer function that receives Events andi starts
 * the acctual analysis (histogram filling) and so on chain.
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

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

void CSCMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) { }

void CSCMonitorModule::getCSCFromMap(int crate, int slot, int& csctype, int& cscposition) {

  int iendcap = -1;
  int istation = -1;
  int iring = -1;

  int id = cscMapping.chamber(iendcap, istation, crate, slot, -1);
  if (id==0) return;
  CSCDetId cid( id );
  iendcap = cid.endcap();
  istation = cid.station();
  iring = cid.ring();
  cscposition = cid.chamber();

  std::string tlabel = getCSCTypeLabel(iendcap, istation, iring );
  std::map<std::string,int>::const_iterator it = tmap.find( tlabel );
  if (it != tmap.end()) {
    csctype = it->second;
  } else {
    csctype = 0;
  }

}  
