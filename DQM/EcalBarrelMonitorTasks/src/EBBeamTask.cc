/*
 * \file EBBeamTask.cc
 *
 * $Date: 2006/02/21 20:32:48 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamTask.h>

EBBeamTask::EBBeamTask(const ParameterSet& ps){

  init_ = false;

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

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    dccMap[dcch.id()] = dcch;

    if ( dccMap[dcch.id()].getRunType() == BEAMH4 ) enable = true;
    if ( dccMap[dcch.id()].getRunType() == BEAMH2 ) enable = true;

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

}

