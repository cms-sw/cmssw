/*
 * \file EBElectronTask.cc
 *
 * $Date: 2006/01/29 17:21:28 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBElectronTask.h>

EBElectronTask::EBElectronTask(const ParameterSet& ps){

  init_ = false;

  // this is a hack, used to fake the EcalBarrel run header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(1) != 4 ) return;

  this->setup();

}

EBElectronTask::~EBElectronTask(){

}

void EBElectronTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBElectronTask::setup(void){

  init_ = true;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBElectronTask");

  }

}

void EBElectronTask::endJob(){

  LogInfo("EBElectronTask") << "analyzed " << ievt_ << " events";

}

void EBElectronTask::analyze(const Event& e, const EventSetup& c){

  // this is a hack, used to fake the EcalBarrel event header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(2) != 4 ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

}

