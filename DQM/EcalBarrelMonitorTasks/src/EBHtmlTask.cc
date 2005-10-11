/*
 * \file EBHtmlTask.cc
 * 
 * $Date: 2005/10/11 16:40:55 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBHtmlTask.h>

EBHtmlTask::EBHtmlTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  local_dbe = dbe;

  logFile.open("EBHtmlTask.log");

}

EBHtmlTask::~EBHtmlTask(){

  logFile.close();

}

void EBHtmlTask::analyze(const edm::Event& e, const edm::EventSetup& c){

//  cout << "EBHtmlTask::analyze" << endl;

  local_dbe->cd("/EcalBarrel/EBPedestalTask/Gain01");

}

