/*
 * \file EBCosmicTask.cc
 * 
 * $Date: 2005/10/30 14:56:49 $
 * $Revision: 1.20 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

EBCosmicTask::EBCosmicTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBCosmicTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask");

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Cut");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude cut SM%02d", i+1);
      meCutMap[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude sel SM%02d", i+1);
      meSelMap[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }
  }

}

EBCosmicTask::~EBCosmicTask(){

  logFile.close();

}

void EBCosmicTask::beginJob(const edm::EventSetup& c){

  ievt = 0;

}

void EBCosmicTask::endJob(){

  cout << "EBCosmicTask: analyzed " << ievt << " events" << endl;

}

void EBCosmicTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EcalUncalibratedRecHitCollection>  hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

//  int nebh = hits->size();
//  cout << "EBCosmicTask: event " << ievt << " hits collection size " << nebh << endl;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    float xval = hit.amplitude();

//    logFile << " hit amplitude " << xval << endl;

    if ( xval >= 500 ) {
      if ( meCutMap[ism-1] ) meCutMap[ism-1]->Fill(xie, xip, xval);
    }

    if ( xval >= 1000 ) {
      if ( meSelMap[ism-1] ) meSelMap[ism-1]->Fill(xie, xip, xval);
    }

  }

}

