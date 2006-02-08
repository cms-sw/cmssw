/*
 * \file DTMonitorModule.cc
 * 
 * $Date: 2005/11/15 14:04:44 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTMonitorModule.h>

#include <DQM/DTMonitorTasks/interface/DTDigiTask.h>
#include <DQM/DTMonitorTasks/interface/DTTestPulsesTask.h>
#include <DQM/DTMonitorTasks/interface/DTLocalRecoTask.h>
#include <DQM/DTMonitorTasks/interface/DTGlobalRecoTask.h>
#include <DQM/DTMonitorTasks/interface/DTLocalTriggerTask.h>


DTMonitorModule::DTMonitorModule(const edm::ParameterSet& ps){

  logFile.open("DTMonitorModule.log");


  outputFile = ps.getUntrackedParameter<string>("outputFile", "");
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  dbe->setVerbose(1);

  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("DriftTubes");
    meStatus  = dbe->bookInt("STATUS");
    meRun     = dbe->bookInt("RUN");
    meEvt     = dbe->bookInt("EVT");
    meRunType = dbe->bookInt("RUNTYPE");
  }

  if ( dbe ) dbe->showDirStructure();

}

DTMonitorModule::~DTMonitorModule(){


  logFile.close();

}

void DTMonitorModule::beginJob(const edm::ParameterSet& ps, const edm::EventSetup& context){

  nevents = 0;

  if ( meStatus ) meStatus->Fill(0);

  
}

void DTMonitorModule::endJob(void) {


  cout << "[DTMonitorModule]: " << nevents << " events analyzed " << endl;

  if ( meStatus ) meStatus->Fill(2);

  if ( outputFile.size() != 0  && dbe ) dbe->save(outputFile);

  // this is to give enough time to the meStatus to reach the clients ...
  sleep(60);

}

void DTMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( meStatus ) meStatus->Fill(1);

  nevents++;

  if ( meEvt ) meEvt->Fill(nevents);



//  sleep(1);

}

