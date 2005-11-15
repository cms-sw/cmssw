/*
 * \file DTMonitorModule.cc
 * 
 * $Date: 2005/11/10 09:08:27 $
 * $Revision: 1.37 $
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

  doDigiTask = ps.getUntrackedParameter<bool>("PerformDigiTask", false);
  doTPTask = ps.getUntrackedParameter<bool>("doTPTask",false);
  doLocalRecoTask = ps.getUntrackedParameter<bool>("doLocalRecoTask",false);
  doGlobalRecoTask = ps.getUntrackedParameter<bool>("doGlobalRecoTask",false); 
  doLocalTriggerTask = ps.getUntrackedParameter<bool>("doLocalTriggerTask",false);

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

  digiTask = new DTDigiTask(ps, dbe);
  tpTask = new DTTestPulsesTask(ps, dbe);
  localRecoTask = new DTLocalRecoTask(ps, dbe);
  globalRecoTask = new DTGlobalRecoTask(ps, dbe);
  localTriggerTask = new DTLocalTriggerTask(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

}

DTMonitorModule::~DTMonitorModule(){

  delete digiTask;
  delete tpTask;
  delete localRecoTask;
  delete globalRecoTask;
  delete localTriggerTask;

  logFile.close();

}

void DTMonitorModule::beginJob(const edm::EventSetup& c){

  nevents = 0;

  if ( meStatus ) meStatus->Fill(0);

  digiTask->beginJob(c);
  tpTask->beginJob(c);
  localRecoTask->beginJob(c);
  globalRecoTask->beginJob(c);
  localTriggerTask->beginJob(c);
  
}

void DTMonitorModule::endJob(void) {

  digiTask->endJob();
  tpTask->endJob();
  localRecoTask->endJob();
  globalRecoTask->endJob();
  localTriggerTask->endJob();

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

  if ( doDigiTask ) digiTask->analyze(e, c);
  if ( doTPTask ) tpTask->analyze(e, c);
  if ( doLocalRecoTask ) localRecoTask->analyze(e, c);
  if ( doGlobalRecoTask ) globalRecoTask->analyze(e, c);
  if ( doLocalTriggerTask ) localTriggerTask->analyze(e, c);


//  sleep(1);

}

