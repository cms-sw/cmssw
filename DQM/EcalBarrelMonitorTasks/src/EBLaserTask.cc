/*
 * \file EBLaserTask.cc
 * 
 * $Date: 2005/10/30 15:34:53 $
 * $Revision: 1.21 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>

EBLaserTask::EBLaserTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBLaserTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBLaserTask");

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser1");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L1", i+1);
      meShapeMapL1[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L1", i+1);
      meAmplMapL1[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L2", i+1);
      meShapeMapL2[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L2", i+1);
      meAmplMapL2[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }
  }

}

EBLaserTask::~EBLaserTask(){

  logFile.close();

}

void EBLaserTask::beginJob(const edm::EventSetup& c){

  ievt = 0;
    
}

void EBLaserTask::endJob(){

  cout << "EBLaserTask: analyzed " << ievt << " events" << endl;

}

void EBLaserTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBLaserTask: event " << ievt << " digi collection size " << nebd << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

//    int ie = id.ieta();
//    int ip = id.iphi();
//    int iz = id.zside();

//    float xie = iz * (ie - 0.5);
//    float xip = ip - 0.5;

    int ism = id.ism();

    int ic = id.ic();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* meShapeMap = 0;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
      }

      if ( ievt >=   1 && ievt <=  600 ) meShapeMap = meShapeMapL1[ism-1];
      if ( ievt >= 601 && ievt <= 1200 ) meShapeMap = meShapeMapL2[ism-1];

      float xval = adc * gain;

      if ( meShapeMap ) meShapeMap->Fill( ic - 0.5, i + 0.5, xval);

    }

  }

  edm::Handle<EcalUncalibratedRecHitCollection>  hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

//  int neh = hits->size();
//  cout << "EBTestPulseTask: event " << ievt << " hits collection size " << neb << endl;

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

    MonitorElement* meAmplMap = 0;

    if ( ievt >=   1 && ievt <=  600 ) meAmplMap = meAmplMapL1[ism-1];
    if ( ievt >= 601 && ievt <= 1200 ) meAmplMap = meAmplMapL2[ism-1];

    float xval = 0.001 * hit.amplitude();

//    logFile << " hit amplitude " << xval << endl;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

  }

}

