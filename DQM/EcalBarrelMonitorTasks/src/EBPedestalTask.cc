/*
 * \file EBPedestalTask.cc
 *
 * $Date: 2006/01/29 17:21:28 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>

EBPedestalTask::EBPedestalTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    mePedMapG01_[i] = 0;
    mePedMapG06_[i] = 0;
    mePedMapG12_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

  // this is a hack, used to fake the EcalBarrel run header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(1) != 2 ) return;

  this->setup();

}

EBPedestalTask::~EBPedestalTask(){

}

void EBPedestalTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBPedestalTask::setup(void){

  init_ = true;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask");

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G01", i+1);
      mePedMapG01_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain06");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G06", i+1);
      mePedMapG06_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPedestalTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPT pedestal SM%02d G12", i+1);
      mePedMapG12_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01", i+1);
      mePnPedMapG01_[i] =  dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16", i+1);
      mePnPedMapG16_[i] =  dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

  }

}

void EBPedestalTask::endJob(){

  LogInfo("EBPedestalTask") << "analyzed " << ievt_ << " events";
}

void EBPedestalTask::analyze(const Event& e, const EventSetup& c){

  // this is a hack, used to fake the EcalBarrel event header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  if ( tmp && tmp->GetBinContent(2) != 2 ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogDebug("EBPedestalTask") << "event " << ievt_ << " digi collection size " << nebd;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    LogDebug("EBPedestalTask") << " det id = " << id;
    LogDebug("EBPedestalTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();

      MonitorElement* mePedMap = 0;

      if ( sample.gainId() == 1 ) mePedMap = mePedMapG12_[ism-1];
      if ( sample.gainId() == 2 ) mePedMap = mePedMapG06_[ism-1];
      if ( sample.gainId() == 3 ) mePedMap = mePedMapG01_[ism-1];

      float xval = float(adc);

      if ( mePedMap ) mePedMap->Fill(xie, xip, xval);

    }

  }

  Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

  int nep = pns->size();
  LogDebug("EBPedestalTask") << "event " << ievt_ << " pns collection size " << nep;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();

//    int ism = id.ism();
    int ism = id.iDCCId();
    int num = id.iPnId();

    LogDebug("EBPedestalTask") << " det id = " << id;
    LogDebug("EBPedestalTask") << " sm, num " << ism << " " << num;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      MonitorElement* mePNPed = 0;

      if ( sample.gainId() == 0 ) mePNPed = mePnPedMapG01_[ism-1];
      if ( sample.gainId() == 1 ) mePNPed = mePnPedMapG16_[ism-1];

      float xval = float(adc);

      if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

    }

  }

}

