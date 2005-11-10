/*
 * \file EBPedOnlineTask.cc
 * 
 * $Date: 2005/11/10 08:32:40 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedOnlineTask.h>

EBPedOnlineTask::EBPedOnlineTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBPedOnlineTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedOnlineTask");

    dbe->setCurrentFolder("EcalBarrel/EBPedOnlineTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal online SM%02d G01", i+1);
      mePedMapG01[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

  }

}

EBPedOnlineTask::~EBPedOnlineTask(){

  logFile.close();

}

void EBPedOnlineTask::beginJob(const edm::EventSetup& c){

  ievt = 0;
    
}

void EBPedOnlineTask::endJob(){

  cout << "EBPedOnlineTask: analyzed " << ievt << " events" << endl;
}

void EBPedOnlineTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBPedOnlineTask: event " << ievt << " digi collection size " << nebd << endl;

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

