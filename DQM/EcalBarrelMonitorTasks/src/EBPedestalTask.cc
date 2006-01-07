/*
 * \file EBPedestalTask.cc
 *
 * $Date: 2006/01/02 12:29:22 $
 * $Revision: 1.24 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPedestalTask.h>

EBPedestalTask::EBPedestalTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBPedestalTask.log");

  for (int i = 0; i < 36 ; i++) {
    mePedMapG01_[i] = 0;
    mePedMapG06_[i] = 0;
    mePedMapG12_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

  Char_t histo[20];

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
      mePnPedMapG01_[i] =  dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16", i+1);
      mePnPedMapG01_[i] =  dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

  }

}

EBPedestalTask::~EBPedestalTask(){

//  logFile_.close();

}

void EBPedestalTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBPedestalTask::endJob(){

  cout << "EBPedestalTask: analyzed " << ievt_ << " events" << endl;
}

void EBPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  edm::Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBPedestalTask: event " << ievt_ << " digi collection size " << nebd << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

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

  edm::Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

//  int nep = pns->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " pns collection size " << nep << endl;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();

//    int ism = id.ism();
    int ism = id.iDCCId();
    int num = id.iPnId();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, num " << ism << " " << num << endl;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      MonitorElement* mePNPed = 0;

      if ( sample.gainId() == 1 ) mePNPed = mePnPedMapG01_[ism-1];
      if ( sample.gainId() == 2 ) mePNPed = mePnPedMapG16_[ism-1];

      float xval = float(adc);

      if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

    }

  }

}

