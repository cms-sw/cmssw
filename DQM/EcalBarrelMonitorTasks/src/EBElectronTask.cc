/*
 * \file EBElectronTask.cc
 *
 * $Date: 2005/12/23 08:57:18 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBElectronTask.h>

EBElectronTask::EBElectronTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBElectronTask.log");

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBElectronTask");

  }

}

EBElectronTask::~EBElectronTask(){

//  logFile_.close();

}

void EBElectronTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBElectronTask::endJob(){

  cout << "EBElectronTask: analyzed " << ievt_ << " events" << endl;
}

void EBElectronTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

}

