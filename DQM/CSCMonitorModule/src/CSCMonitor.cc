#include <memory>
#include <iostream>


#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//#include "PluginManager/PluginManager.h"

using namespace std;

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig ){

 printout=false;
 
 nEvents=0;
 dduBX=0;
 L1ANumber=0;
 FEBUnpacked =0;
 
 dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  meCollection[0] = book_common();
  int chID=256; //temporary dummy value
  meCollection[chID] = book_chamber(chID);


  dbe->showDirStructure();
}



CSCMonitor::~CSCMonitor()
{
   
  
}


void CSCMonitor::process(CSCDCCEventData & dccData )
{
   
      
   const vector<CSCDDUEventData> & dduData = dccData.dduData(); 

   for (int ddu=0; ddu<(int)dduData.size(); ++ddu) { 
   
         MonitorDDU(dduData[ddu]);
   
      }
  
}






#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<MonitorInterface,CSCMonitor> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(MonitorInterface,maker)



