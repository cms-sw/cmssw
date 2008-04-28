/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitor.cc
 *
 *    Description:  Backward Compatible Object
 *
 *        Version:  1.0
 *        Created:  04/28/2008 09:51:42 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
<<<<<<< CSCMonitor.cc
=======
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig )
{

  setParameters();

  printout = iConfig.getUntrackedParameter<bool>("monitorVerbosity", false);
  xmlHistosBookingCfgFile = iConfig.getUntrackedParameter<std::string>("BookingFile", "emuDQMBooking.xml"); 
  fSaveHistos  = iConfig.getUntrackedParameter<bool>("CSCDQMSaveRootFile", false);
  saveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EventsInterval", 20000);
  RootHistoFile  = iConfig.getUntrackedParameter<std::string>("RootFileName", "DQM_CSC_Monitor.root");

  // CSC Mapping
  cscMapping  = CSCReadoutMappingFromFile(iConfig);
      
  this->loadBooking();
>>>>>>> 1.20

CSCMonitor::CSCMonitor(const edm::ParameterSet& ps) {
  mm = edm::Service<CSCMonitorModule>().operator->();
}

CSCMonitor::~CSCMonitor() {
  delete mm;
}

void CSCMonitor::process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData) {
  mm->process(examiner, dccData);
}

