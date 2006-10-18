/*
 * \file DTLocalTriggerTask.cc
 * 
 * $Date: 2006/02/15 08:24:56 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTLocalTriggerTask.h>

#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

using namespace std;

DTLocalTriggerTask::DTLocalTriggerTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
				       const edm::EventSetup& context){

  logFile.open("DTLocalTriggerTask.log");


  if ( dbe ) {
    dbe->setCurrentFolder("DT/DTLocalTriggerTask");


  }

}

DTLocalTriggerTask::~DTLocalTriggerTask(){

  logFile.close();

}

void DTLocalTriggerTask::beginJob(const edm::EventSetup& c){

  nevents = 0;

}

void DTLocalTriggerTask::endJob(){

  cout << "DTLocalTriggerTask: analyzed " << nevents << " events" << endl;

}

void DTLocalTriggerTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;


}

