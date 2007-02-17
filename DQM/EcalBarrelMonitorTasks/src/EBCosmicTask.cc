/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2007/02/17 17:04:43 $
 * $Revision: 1.61 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBCosmicTask::EBCosmicTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrumMap_[i] = 0;
  }

}

EBCosmicTask::~EBCosmicTask(){

}

void EBCosmicTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");
    dbe->rmdir("EcalBarrel/EBCosmicTask");
  }

}

void EBCosmicTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy cut SM%02d", i+1);
      meCutMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy sel SM%02d", i+1);
      meSelMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy spectrum SM%02d", i+1);
      meSpectrumMap_[i] = dbe->book1D(histo, histo, 100, 0., 5.);
    }

  }

}

void EBCosmicTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      if ( meCutMap_[i] ) dbe->removeElement( meCutMap_[i]->getName() );
      meCutMap_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      if ( meSelMap_[i] ) dbe->removeElement( meSelMap_[i]->getName() );
      meSelMap_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      if ( meSpectrumMap_[i] ) dbe->removeElement( meSpectrumMap_[i]->getName() );
      meSpectrumMap_[i] = 0;
    }

  }

  init_ = false;

}

void EBCosmicTask::endJob(void){

  LogInfo("EBCosmicTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBCosmicTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(dcch.id());
    if ( i != dccMap.end() ) continue;

    dccMap[dcch.id()] = dcch;

    if ( dcch.getRunType() == EcalDCCHeaderBlock::COSMIC ||
         dcch.getRunType() == EcalDCCHeaderBlock::MTCC ) enable = true;

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRecHitCollection> hits;
  e.getByLabel("ecalRecHitMaker", "EcalRecHitsEB", hits);

  int nebh = hits->size();
  LogDebug("EBCosmicTask") << "event " << ievt_ << " hits collection size " << nebh;

  for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

    if ( ! ( dccMap[ism-1].getRunType() == EcalDCCHeaderBlock::COSMIC ||
             dccMap[ism-1].getRunType() == EcalDCCHeaderBlock::MTCC ) ) continue;

    LogDebug("EBCosmicTask") << " det id = " << id;
    LogDebug("EBCosmicTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    float xval = hit.energy();
    if ( xval <= 0. ) xval = 0.0;

    LogDebug("EBCosmicTask") << " hit energy " << xval;

    const float lowThreshold = 0.25;
    const float highThreshold = 0.50;

    if ( xval >= lowThreshold ) {
      if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xie, xip, xval);
    }

    if ( xval >= highThreshold ) {
      if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xie, xip, xval);
    }

    if ( meSpectrumMap_[ism-1] ) meSpectrumMap_[ism-1]->Fill(xval);

  }

}

