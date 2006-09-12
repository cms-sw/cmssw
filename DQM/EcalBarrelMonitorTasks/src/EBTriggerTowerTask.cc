/*
 * \file EBTriggerTowerTask.cc
 *
 * $Date: 2006/09/12 14:58:49 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBTriggerTowerTask.h>

EBTriggerTowerTask::EBTriggerTowerTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    meEtMap_[i] = 0;
    meVeto_[i] = 0;
    meFlags_[i] = 0;
    for (int j = 0; j < 68 ; j++) {
      meEtMapT_[i][j] = 0;
      meEtMapR_[i][j] = 0;
    }
  }

}

EBTriggerTowerTask::~EBTriggerTowerTask(){

}

void EBTriggerTowerTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTriggerTowerTask");
    dbe->rmdir("EcalBarrel/EBTriggerTowerTask");
  }

}

void EBTriggerTowerTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTriggerTowerTask");

    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTTT Et map SM%02d", i+1);
      meEtMap_[i] = dbe->bookProfile2D(histo, histo, 17, 0., 17., 4, 0., 4., 4096, 0., 4096., "s");
      sprintf(histo, "EBTTT FineGrainVeto SM%02d", i+1);
      meVeto_[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      sprintf(histo, "EBTTT Flags SM%02d", i+1);
      meFlags_[i] = dbe->book3D(histo, histo, 17, 0., 17., 4, 0., 4., 7, 0., 7.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBTriggerTowerTask/EnergyMaps");

    for (int i = 0; i < 36 ; i++) {
      for (int j = 0; j < 68 ; j++) {
        sprintf(histo, "EBTTT Et T SM%02d TT%02d", i+1, j+1);
        meEtMapT_[i][j] = dbe->book1D(histo, histo, 4096, 0., 4096.);
        sprintf(histo, "EBTTT Et R SM%02d TT%02d", i+1, j+1);
        meEtMapR_[i][j] = dbe->book1D(histo, histo, 4096, 0., 4096.);
      }
    }

  }

}

void EBTriggerTowerTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTriggerTowerTask");

    for ( int i = 0; i < 36; i++ ) {
      if ( meEtMap_[i] ) dbe->removeElement( meEtMap_[i]->getName() );
      meEtMap_[i] = 0;
      if ( meVeto_[i] ) dbe->removeElement( meVeto_[i]->getName() );
      meVeto_[i] = 0; 
      if ( meFlags_[i] ) dbe->removeElement( meFlags_[i]->getName() );
      meFlags_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBTriggerTowerTask/EnergyMaps");

    for ( int i = 0; i < 36; i++ ) {
      for ( int j = 0; j < 36; j++ ) {
        if ( meEtMapT_[i][j] ) dbe->removeElement( meEtMapT_[i][j]->getName() );
        meEtMapT_[i][j] = 0;
        if ( meEtMapR_[i][j] ) dbe->removeElement( meEtMapR_[i][j]->getName() );
        meEtMapR_[i][j] = 0;
      }      
    }

  }

  init_ = false;

}

void EBTriggerTowerTask::endJob(void){

  LogInfo("EBTriggerTowerTask") << "analyzed " << ievt_ << " events";

  this->cleanup();

}

void EBTriggerTowerTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalTrigPrimDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogDebug("EBTriggerTowerTask") << "event " << ievt_ << " digi collection size " << nebd;

  for ( EcalTrigPrimDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EcalTriggerPrimitiveDigi data = (*digiItr);
    EcalTrigTowerDetId id = data.id();

    int iet = id.ieta();
    int ipt = id.iphi();

    int ismt = id.iDCC();

    float xiet = iet - 0.5;
    float xipt = ipt - 0.5;

    LogDebug("EBTriggerTowerTask") << " det id = " << id;
    LogDebug("EBTriggerTowerTask") << " sm, eta, phi " << ismt << " " << iet << " " << ipt;

  }

}

