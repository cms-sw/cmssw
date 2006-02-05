/*
 * \file EBPedestalOnlineTask.cc
 *
 * $Date: 2006/01/29 17:21:28 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalOnlineTask.h>

EBPedestalOnlineTask::EBPedestalOnlineTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    mePedMapG12_[i] = 0;
  }

  this->setup();

}

EBPedestalOnlineTask::~EBPedestalOnlineTask(){

}

void EBPedestalOnlineTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBPedestalOnlineTask::setup(void){

  init_ = true;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedestalOnlineTask");

    dbe->setCurrentFolder("EcalBarrel/EBPedestalOnlineTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPOT pedestal SM%02d G12", i+1);
      mePedMapG12_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

  }

}

void EBPedestalOnlineTask::endJob(){

  LogInfo("EBPedestalOnlineTask") << "analyzed " << ievt_ << " events";
}

void EBPedestalOnlineTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogDebug("EBPedestalOnlineTask") << "event " << ievt_ << " digi collection size " << nebd;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    LogDebug("EBPedestalOnlineTask") << " det id = " << id;
    LogDebug("EBPedestalOnlineTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    for (int i = 0; i < 3; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();

      MonitorElement* mePedMap = 0;

      if ( sample.gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
      if ( sample.gainId() == 2 ) mePedMap = 0;
      if ( sample.gainId() == 3 ) mePedMap = 0;

      float xval = float(adc);

      if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

    }

  }

}

