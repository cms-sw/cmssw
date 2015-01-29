/*
 * \file DTDigiTask.cc
 *
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTGlobalRecoTask.h>
#include "DQMServices/Core/interface/DQMStore.h"


using namespace std;

DTGlobalRecoTask::DTGlobalRecoTask(const edm::ParameterSet& ps, const edm::EventSetup& context){
 logFile.open("DTGlobalRecoTask.log");
 nevents = 0;
}

DTGlobalRecoTask::~DTGlobalRecoTask(){
  cout << "DTGlobalRecoTask: analyzed " << nevents << " events" << endl;
  logFile.close();
}


void DTGlobalRecoTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & context) {
    ibooker.setCurrentFolder("DT/DTGlobalRecoTask");
}

void DTGlobalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  nevents++;
}


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
