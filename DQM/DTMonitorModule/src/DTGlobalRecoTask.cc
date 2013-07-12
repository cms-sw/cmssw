/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2008/03/01 00:39:54 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTGlobalRecoTask.h>
#include "DQMServices/Core/interface/DQMStore.h"


using namespace std;

DTGlobalRecoTask::DTGlobalRecoTask(const edm::ParameterSet& ps, DQMStore* dbe,
				   const edm::EventSetup& context){

  logFile.open("DTGlobalRecoTask.log");


  if ( dbe ) {
    dbe->setCurrentFolder("DT/DTGlobalRecoTask");


  }

}

DTGlobalRecoTask::~DTGlobalRecoTask(){

  logFile.close();

}

void DTGlobalRecoTask::beginJob(){

  nevents = 0;

}

void DTGlobalRecoTask::endJob(){

  cout << "DTGlobalRecoTask: analyzed " << nevents << " events" << endl;

}

void DTGlobalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;




}

