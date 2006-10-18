/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/02/15 08:24:56 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTGlobalRecoTask.h>

#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

using namespace std;

DTGlobalRecoTask::DTGlobalRecoTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
				   const edm::EventSetup& context){

  logFile.open("DTGlobalRecoTask.log");


  if ( dbe ) {
    dbe->setCurrentFolder("DT/DTGlobalRecoTask");


  }

}

DTGlobalRecoTask::~DTGlobalRecoTask(){

  logFile.close();

}

void DTGlobalRecoTask::beginJob(const edm::EventSetup& c){

  nevents = 0;

}

void DTGlobalRecoTask::endJob(){

  cout << "DTGlobalRecoTask: analyzed " << nevents << " events" << endl;

}

void DTGlobalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;




}

