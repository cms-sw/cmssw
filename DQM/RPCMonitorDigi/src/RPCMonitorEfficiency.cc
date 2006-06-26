/** \file
 *
 *  implementation of RPCMonitorEfficiency class
 *
 *  $Date: 2006/06/23 12:29:22 $
 *  $Revision: 1.8 $
 *
 * \author Ilaria Segoni
 */

#include <map>
#include <string>

#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>



///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


RPCMonitorEfficiency::RPCMonitorEfficiency( const edm::ParameterSet& pset ):counter(0){

  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPCEfficiency");
  saveRootFile  = pset.getUntrackedParameter<bool>("EfficDQMSaveRootFile", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EfficEventsInterval", 10000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameEfficiency", "RPCMonitorEfficiency.root"); 
  
  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();


}


RPCMonitorEfficiency::~RPCMonitorEfficiency(){
}

void RPCMonitorEfficiency::endJob(void)
{
  dbe->save(RootFileName);  
}


void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup ){
 edm::LogInfo (nameInLog) <<"Beginning event efficiency evaluation " << counter;


  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) {
    dbe->save(RootFileName);
  }
  
  counter++;
}
 
 

