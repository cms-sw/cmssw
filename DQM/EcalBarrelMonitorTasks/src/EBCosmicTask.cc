/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2006/04/07 09:34:02 $
 * $Revision: 1.44 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

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

}

void EBCosmicTask::setup(void){

  init_ = true;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy cut SM%02d", i+1);
      meCutMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy sel SM%02d", i+1);
      meSelMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT energy spectrum SM%02d", i+1);
      meSpectrumMap_[i] = dbe->book1D(histo, histo, 100, 0., 250.);
    }

  }

}

void EBCosmicTask::endJob(){

  LogInfo("EBCosmicTask") << "analyzed " << ievt_ << " events";

}

void EBCosmicTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    dccMap[dcch.id()] = dcch;

    if ( dccMap[dcch.id()].getRunType() == EcalDCCHeaderBlock::COSMIC ) enable = true;

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRecHitCollection> hits;
  e.getByLabel("ecalRecHitMaker", "EcalRecHits", hits);

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

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::COSMIC ) continue;

    LogDebug("EBCosmicTask") << " det id = " << id;
    LogDebug("EBCosmicTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    float xval = hit.energy();

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

