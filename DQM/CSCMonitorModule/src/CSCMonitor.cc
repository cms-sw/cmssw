/** \file
 * 
 *  implementation of CSCMonitor class
 *
 *  $Date: 2006/08/30 13:11:09 $
 *  $Revision: 1.15 $
 *
 * \author Ilaria Segoni
 */

#include <memory>
#include <iostream>


#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"


#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig ){
 
 
 printout = iConfig.getUntrackedParameter<bool>("monitorVerbosity", false);
 saveRootFile  = iConfig.getUntrackedParameter<bool>("CSCDQMSaveRootFile", false); 
 saveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EventsInterval", 10000); 
 RootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName", "CSCMonitor.root"); 

 for(int ddu=0; ddu<maxDDU; ddu++) {
   dduBooked[ddu]=false;
   dduBX[ddu]=0;
   L1ANumber[ddu]=0;
 }
 for(int cmb=0; cmb<maxCMBID; cmb++) cmbBooked[cmb]=false;

 nEvents=0;
 FEBUnpacked =0;
 
 dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();
}



CSCMonitor::~CSCMonitor()
{
   
  
}


void CSCMonitor::process(CSCDCCEventData & dccData )
{
   
  nEvents = nEvents +1;
        
   const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 

  //edm::LogInfo ("CSC DQM ") << "CSCMonitor::process #" << dec << nEvents 
      //       << "> Number of DDU = " <<dduData.size();
        

   for (int ddu=0; ddu<(int)dduData.size(); ++ddu) { 
   
         this->MonitorDDU(dduData[ddu], ddu );
   
      }
      
      
  if((!(nEvents%saveRootFileEventsInterval ))&&(saveRootFile ) ) {
    dbe->save(RootFileName);
  }
   
 // usleep(100000);
}






#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<CSCMonitorInterface,CSCMonitor> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(CSCMonitor,maker);



