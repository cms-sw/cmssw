/*
 * \file EBPedestalTask.cc
 * 
 * $Date: 2005/10/11 17:55:11 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>

EBPedestalTask::EBPedestalTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBPedestalTask.log");

  ievt = 0;

  Char_t histo[20];

  dbe->setCurrentFolder("EcalBarrel/EBPedestalTask");

  dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain01");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G01", i+1);
    mePedMapG01[i] = dbe->bookProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 4096, 0., 4096.);
  }

  dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain06");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G06", i+1);
    mePedMapG06[i] = dbe->bookProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 4096, 0., 4096.);
  }

  dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain12");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G12", i+1);
    mePedMapG12[i] = dbe->bookProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 4096, 0., 4096.);
  }

}

EBPedestalTask::~EBPedestalTask(){

  logFile.close();

  cout << "EBPedestalTask: analyzed " << ievt << " events" << endl;
}

void EBPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int neb = digis->size();
//  cout << "EBPedestalTask: event " << ievt << " collection size " << neb << endl;

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

      if ( i <= 3 ) {
        if ( mePedMap ) mePedMap->Fill(xip, xie, xval);
      }

// only if the event is a pedestal event
//
//      if ( i >= 4 ) {
//        if ( mePedMap ) mePedMap->Fill(xip, xie, xval);
//      }

    }

  }

}

