/*
 * \file EBElectronTask.cc
 *
 * $Date: 2005/12/30 10:24:29 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBElectronTask.h>

EBElectronTask::EBElectronTask(const edm::ParameterSet& ps){

//  logFile_.open("EBElectronTask.log");

  init_ = false;

  // this is a hack, used to fake the EcalBarrel run header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(1) != 4 ) return;

  this->setup();

}

EBElectronTask::~EBElectronTask(){

//  logFile_.close();

}

void EBElectronTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBElectronTask::setup(void){

  init_ = true;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBElectronTask");

  }

}

void EBElectronTask::endJob(){

  cout << "EBElectronTask: analyzed " << ievt_ << " events" << endl;
}

void EBElectronTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  // this is a hack, used to fake the EcalBarrel event header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(2) != 4 ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

}

