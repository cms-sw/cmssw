/*
 * \file EBCosmicTask.cc
 *
 * $Date: 2006/01/05 08:59:20 $
 * $Revision: 1.32 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

EBCosmicTask::EBCosmicTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBCosmicTask.log");

  for (int i = 0; i < 36 ; i++) {
    meCutMap_[i] = 0;
    meSelMap_[i] = 0;
    meSpectrumMap_[i] = 0;
  }

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude cut SM%02d", i+1);
      meCutMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude sel SM%02d", i+1);
      meSelMap_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Spectrum");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude spectrum SM%02d", i+1);
      meSpectrumMap_[i] = dbe->book1D(histo, histo, 100, 0., 1000.);
    }

  }

}

EBCosmicTask::~EBCosmicTask(){

//  logFile_.close();

}

void EBCosmicTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBCosmicTask::endJob(){

  cout << "EBCosmicTask: analyzed " << ievt_ << " events" << endl;

}

void EBCosmicTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  edm::Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

//  int nebh = hits->size();
//  cout << "EBCosmicTask: event " << ievt_ << " hits collection size " << nebh << endl;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;

    float xval = hit.amplitude();

//    logFile_ << " hit amplitude " << xval << endl;

    const float lowThreshold = 5.;
    const float highThreshold = 10.;

    if ( xval >= lowThreshold ) {
      if ( meCutMap_[ism-1] ) meCutMap_[ism-1]->Fill(xie, xip, xval);
    }

    if ( xval >= highThreshold ) {
      if ( meSelMap_[ism-1] ) meSelMap_[ism-1]->Fill(xie, xip, xval);
    }

    if ( meSpectrumMap_[ism-1] ) meSpectrumMap_[ism-1]->Fill(xval);

  }

}

