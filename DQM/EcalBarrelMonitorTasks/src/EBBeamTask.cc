/*
 * \file EBBeamTask.cc
 *
 * $Date: 2006/02/05 22:19:22 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamTask.h>

EBBeamTask::EBBeamTask(const ParameterSet& ps){

  init_ = false;

  // this is a hack, used to fake the EcalBarrel run header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(1) != 4 ) return;

  this->setup();

}

EBBeamTask::~EBBeamTask(){

}

void EBBeamTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBBeamTask::setup(void){

  init_ = true;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamTask");

  }

}

void EBBeamTask::endJob(){

  LogInfo("EBBeamTask") << "analyzed " << ievt_ << " events";

}

void EBBeamTask::analyze(const Event& e, const EventSetup& c){

  // this is a hack, used to fake the EcalBarrel event header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(2) != 4 ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

}

