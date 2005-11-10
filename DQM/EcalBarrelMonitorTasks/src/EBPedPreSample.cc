/*
 * \file EBPedPreSampleTask.cc
 * 
 * $Date: 2005/11/10 09:08:27 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedPreSampleTask.h>

EBPedPreSampleTask::EBPedPreSampleTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBPedPreSampleTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedPreSampleTask");

    dbe->setCurrentFolder("EcalBarrel/EBPedPreSampleTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal PreSample SM%02d G01", i+1);
      mePedMapG01[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

  }

}

EBPedPreSampleTask::~EBPedPreSampleTask(){

  logFile.close();

}

void EBPedPreSampleTask::beginJob(const edm::EventSetup& c){

  ievt = 0;
    
}

void EBPedPreSampleTask::endJob(){

  cout << "EBPedPreSampleTask: analyzed " << ievt << " events" << endl;
}

void EBPedPreSampleTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBPedPreSampleTask: event " << ievt << " digi collection size " << nebd << endl;

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

    for (int i = 0; i < 3; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* mePedMap = 0;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
        mePedMap = 0;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
        mePedMap = 0;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
        mePedMap = mePedMapG01[ism-1];
      }

      float xval = adc * gain;

      if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

    }

  }

}

