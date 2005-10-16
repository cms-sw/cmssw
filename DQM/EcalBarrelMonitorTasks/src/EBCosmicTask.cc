/*
 * \file EBCosmicTask.cc
 * 
 * $Date: 2005/10/16 12:20:27 $
 * $Revision: 1.12 $
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
      meCutMap[i] = dbe->bookProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBCosmicTask/Sel");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBCT amplitude sel SM%02d", i+1);
      meSelMap[i] = dbe->bookProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 4096, 0., 4096.);
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

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int neb = digis->size();
//  cout << "EBCosmicTask: event " << ievt << " collection size " << neb << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

    logFile << " det id = " << id << endl;
    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    float xped = 0.;

    float xvalmax = 0.;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
      }

      float xval = adc * gain;

// use the 3 first samples for the pedestal

      if ( i <= 2 ) {
        xped = xped + xval / 3.;
      }

// average rms per crystal

      float xrms = 1.2;

// signal samples

      if ( i >= 3 ) {
        xval = xval - xped;
        if ( xval >= xrms && xval >= xvalmax ) xvalmax = xval;
      }

    }

    if ( xvalmax >= 5 ) {
      if ( meCutMap[ism-1] ) meCutMap[ism-1]->Fill(xip, xie, xvalmax);
    }

    if ( xvalmax >= 10 ) {
      if ( meSelMap[ism-1] ) meSelMap[ism-1]->Fill(xip, xie, xvalmax);
    }

  }

}

