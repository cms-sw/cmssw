#include <memory>
#include <iostream>


#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// scommenta #include "EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h"

using namespace std;

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig ){

 nEvents=0;
 dduBX=0;
 L1ANumber=0;
 
 dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  meCollection[0] = book_common();
//  int chID=256;
//  cout<<"about to book chamber plots"<<endl;
//  meCollection[chID] = book_chamber(chID);


  dbe->showDirStructure();
}



CSCMonitor::~CSCMonitor()
{
   
  
}

/*
void CSCMonitor::process(CSCDCCUnpacker & unpacker)
{
   
  dduEvent=unpacker.dduEventData;
  MonitorDDU();
  
  
  
}

*/







//define this as a plug-in
//DEFINE_FWK_MODULE(CSCMonitor)

