/*
 * \file DTLocalRecoTask.cc
 * 
 * $Date: 2006/02/06 19:20:22 $
 * $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTLocalRecoTask.h>

#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>


DTLocalRecoTask::DTLocalRecoTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
				 const edm::EventSetup& context){

  logFile.open("DTLocalRecoTask.log");


  if ( dbe ) {
    dbe->setCurrentFolder("DT/DTLocalRecoTask");


  }

}

DTLocalRecoTask::~DTLocalRecoTask(){

  logFile.close();

}

void DTLocalRecoTask::beginJob(const edm::EventSetup& c){

  nevents = 0;

}

void DTLocalRecoTask::endJob(){

  cout << "DTLocalRecoTask: analyzed " << nevents << " events" << endl;

}

void DTLocalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;


}

