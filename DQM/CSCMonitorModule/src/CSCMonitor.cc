#include <memory>
#include <iostream>


#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"


#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using namespace std;

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig ){

 printout=true;
 for(int ddu=0; ddu<maxDDU; ddu++) dduBooked[ddu]=false;
 for(int cmb=0; cmb<maxCMBID; cmb++) cmbBooked[cmb]=false;

 nEvents=0;
 dduBX=0;
 L1ANumber=0;
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
        
   const vector<CSCDDUEventData> & dduData = dccData.dduData(); 

  if(printout) cout << "CSCMonitor::process #" << dec << nEvents 
             << "> Number of DDU = " <<dduData.size()<<endl;
        

   for (int ddu=0; ddu<(int)dduData.size(); ++ddu) { 
   
         MonitorDDU(dduData[ddu], ddu );
   
      }
  usleep(100000);
}






#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<CSCMonitorInterface,CSCMonitor> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(CSCMonitor,maker)



