/*
 * \file EBPedestalTask.cc
 * 
 * $Date: 2005/10/29 09:48:14 $
 * $Revision: 1.15 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>

EBPedestalTask::EBPedestalTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBPedestalTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask");

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G01", i+1);
      mePedMapG01[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain06");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G06", i+1);
      mePedMapG06[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G12", i+1);
      mePedMapG12[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }
  }

}

EBPedestalTask::~EBPedestalTask(){

  logFile.close();

}

void EBPedestalTask::beginJob(const edm::EventSetup& c){

  ievt = 0;
    
}

void EBPedestalTask::endJob(){

  cout << "EBPedestalTask: analyzed " << ievt << " events" << endl;
}

void EBPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int ned = digis->size();
//  cout << "EBPedestalTask: event " << ievt << " digi collection size " << neb << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* mePedMap = 0;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
        mePedMap = mePedMapG12[ism-1];
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
        mePedMap = mePedMapG06[ism-1];
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
        mePedMap = mePedMapG01[ism-1];
      }

      float xval = adc * gain;

// generic event: first 3 samples, 0 to 2

      if ( i <= 2 ) {
        if ( mePedMap ) mePedMap->Fill(xie, xip, xval);
      }

// pedestal event: last 7 samples, 3 to 9
//
//      if ( i >= 3 ) {
//        if ( mePedMap ) mePedMap->Fill(xie, xip, xval);
//      }

    }

  }

}

