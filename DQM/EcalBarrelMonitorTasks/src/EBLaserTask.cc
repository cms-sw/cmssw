/*
 * \file EBLaserTask.cc
 *
 * $Date: 2006/01/07 11:46:49 $
 * $Revision: 1.34 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>

EBLaserTask::EBLaserTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile.open("EBLaserTask.log");

  for (int i = 0; i < 36 ; i++) {
    meShapeMapL1_[i] = 0;
    meAmplMapL1_[i] = 0;
    meAmplPNMapL1_[i] = 0;
    mePnAmplMapG01L1_[i] = 0;
    mePnPedMapG01L1_[i] = 0;
    mePnAmplMapG16L1_[i] = 0;
    mePnPedMapG16L1_[i] = 0;
    meShapeMapL2_[i] = 0;
    meAmplMapL2_[i] = 0;
    meAmplPNMapL2_[i] = 0;
    mePnAmplMapG01L2_[i] = 0;
    mePnPedMapG01L2_[i] = 0;
    mePnAmplMapG16L2_[i] = 0;
    mePnPedMapG16L2_[i] = 0;
    meShapeMapL3_[i] = 0;
    meAmplMapL3_[i] = 0;
    meAmplPNMapL3_[i] = 0;
    mePnAmplMapG01L3_[i] = 0;
    mePnPedMapG01L3_[i] = 0;
    mePnAmplMapG16L3_[i] = 0;
    mePnPedMapG16L3_[i] = 0;
    meShapeMapL4_[i] = 0;
    meAmplMapL4_[i] = 0;
    meAmplPNMapL4_[i] = 0;
    mePnAmplMapG01L4_[i] = 0; 
    mePnPedMapG01L4_[i] = 0;
    mePnAmplMapG16L4_[i] = 0;
    mePnPedMapG16L4_[i] = 0;
  }

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBLaserTask");

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser1");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L1", i+1);
      meShapeMapL1_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L1", i+1);
      meAmplMapL1_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude over PN SM%02d L1", i+1);
      meAmplPNMapL1_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L2", i+1);
      meShapeMapL2_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L2", i+1);
      meAmplMapL2_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude over PN SM%02d L2", i+1);
      meAmplPNMapL2_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L3", i+1);
      meShapeMapL3_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L3", i+1);
      meAmplMapL3_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude over PN SM%02d L3", i+1);
      meAmplPNMapL3_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L4", i+1);
      meShapeMapL4_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude SM%02d L4", i+1);
      meAmplMapL4_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBLT amplitude over PN SM%02d L4", i+1);
      meAmplPNMapL4_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L1", i+1);
      mePnAmplMapG01L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L1", i+1);
      mePnPedMapG01L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L1", i+1);
      mePnAmplMapG16L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L1", i+1);
      mePnPedMapG16L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L2", i+1);
      mePnAmplMapG01L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L2", i+1);
      mePnPedMapG01L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L2", i+1);
      mePnAmplMapG16L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L2", i+1);
      mePnPedMapG16L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L3", i+1);
      mePnAmplMapG01L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L3", i+1);
      mePnPedMapG01L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L3", i+1);
      mePnAmplMapG16L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L3", i+1);
      mePnPedMapG16L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L4", i+1);
      mePnAmplMapG01L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L4", i+1);
      mePnPedMapG01L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L4", i+1);
      mePnAmplMapG16L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L4", i+1);
      mePnPedMapG16L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

  }

}

EBLaserTask::~EBLaserTask(){

//  logFile.close();

}

void EBLaserTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBLaserTask::endJob(){

  cout << "EBLaserTask: analyzed " << ievt_ << " events" << endl;

}

void EBLaserTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  edm::Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBLaserTask: event " << ievt_ << " digi collection size " << nebd << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

//    int ie = id.ieta();
//    int ip = id.iphi();

//    float xie = ie - 0.5;
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

      if ( sample.gainId() == 1 ) gain = 1./12.;
      if ( sample.gainId() == 2 ) gain = 1./ 6.;
      if ( sample.gainId() == 3 ) gain = 1./ 1.;

      if ( ievt_ >=    1 && ievt_ <=  600 ) meShapeMap = meShapeMapL1_[ism-1];
      if ( ievt_ >=  601 && ievt_ <= 1200 ) meShapeMap = meShapeMapL1_[ism-1];
      if ( ievt_ >= 1201 && ievt_ <= 1800 ) meShapeMap = meShapeMapL2_[ism-1];
      if ( ievt_ >= 1801 && ievt_ <= 2400 ) meShapeMap = meShapeMapL2_[ism-1];

      float xval = float(adc) * gain;

      if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

    }

  }

  edm::Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

  float adc[36];

  for ( int i = 0; i < 36; i++ ) adc[i] = 0.;

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

    float xvalped = 0.;

    for (int i = 0; i < 4; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* mePNPed = 0;

      if ( sample.gainId() == 0 ) {
        gain = 1./ 1.;
        if ( ievt_ >=    1 && ievt_ <=  600 ) mePNPed = mePnPedMapG01L1_[ism-1];
        if ( ievt_ >=  601 && ievt_ <= 1200 ) mePNPed = mePnPedMapG01L1_[ism-1];
        if ( ievt_ >= 1201 && ievt_ <= 1800 ) mePNPed = mePnPedMapG01L2_[ism-1];
        if ( ievt_ >= 1801 && ievt_ <= 2400 ) mePNPed = mePnPedMapG01L2_[ism-1];
      }
      if ( sample.gainId() == 1 ) {
        gain = 1./16.;
        if ( ievt_ >=    1 && ievt_ <=  600 ) mePNPed = mePnPedMapG16L1_[ism-1];
        if ( ievt_ >=  601 && ievt_ <= 1200 ) mePNPed = mePnPedMapG16L1_[ism-1];
        if ( ievt_ >= 1201 && ievt_ <= 1800 ) mePNPed = mePnPedMapG16L2_[ism-1];
        if ( ievt_ >= 1801 && ievt_ <= 2400 ) mePNPed = mePnPedMapG16L2_[ism-1];
      }

      float xval = float(adc) * gain;

      if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

      xvalped = xvalped + xval;

    }

    xvalped = xvalped / 4;

    float xvalmax = 0.;
    int gainIdmax = 0;

    MonitorElement* mePN = 0;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();
      float gain = 0.;

      if ( sample.gainId() == 0 ) gain = 1./ 1.;
      if ( sample.gainId() == 1 ) gain = 1./16.;

      float xval = float(adc) * gain;

      if ( xval >= xvalmax ) {
        xvalmax = xval;
        gainIdmax = sample.gainId();
      }

    }

    xvalmax = xvalmax - xvalped;

    if ( gainIdmax == 0 ) {
      if ( ievt_ >=    1 && ievt_ <=  600 ) mePN = mePnAmplMapG01L1_[ism-1];
      if ( ievt_ >=  601 && ievt_ <= 1200 ) mePN = mePnAmplMapG01L1_[ism-1];
      if ( ievt_ >= 1201 && ievt_ <= 1800 ) mePN = mePnAmplMapG01L2_[ism-1];
      if ( ievt_ >= 1801 && ievt_ <= 2400 ) mePN = mePnAmplMapG01L2_[ism-1];
    }
    if ( gainIdmax == 1 ) {
      if ( ievt_ >=    1 && ievt_ <=  600 ) mePN = mePnAmplMapG16L1_[ism-1];
      if ( ievt_ >=  601 && ievt_ <= 1200 ) mePN = mePnAmplMapG16L1_[ism-1];
      if ( ievt_ >= 1201 && ievt_ <= 1800 ) mePN = mePnAmplMapG16L2_[ism-1];
      if ( ievt_ >= 1801 && ievt_ <= 2400 ) mePN = mePnAmplMapG16L2_[ism-1];
    }

    if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

    if ( num == 1 ) adc[ism-1] = xvalmax;

  }

  edm::Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

//  int neh = hits->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " hits collection size " << neh << endl;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    MonitorElement* meAmplMap = 0;
    MonitorElement* meAmplPNMap = 0;

    if ( ievt_ >=    1 && ievt_ <=  600 ) {
      meAmplMap = meAmplMapL1_[ism-1];
      meAmplPNMap = meAmplPNMapL1_[ism-1];
    }
    if ( ievt_ >=  601 && ievt_ <= 1200 ) {
      meAmplMap = meAmplMapL1_[ism-1];
      meAmplPNMap = meAmplPNMapL1_[ism-1];
    }
    if ( ievt_ >= 1201 && ievt_ <= 1800 ) {
      meAmplMap = meAmplMapL2_[ism-1];
      meAmplPNMap = meAmplPNMapL2_[ism-1];
    }
    if ( ievt_ >= 1801 && ievt_ <= 2400 ) {
      meAmplMap = meAmplMapL2_[ism-1];
      meAmplPNMap = meAmplPNMapL2_[ism-1];
    }

    float xval = hit.amplitude();

//    logFile << " hit amplitude " << xval << endl;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

    float yval = 0.;
    if ( adc[ism-1] != 0. ) yval = xval / adc[ism-1];

//    logFile << " hit amplitude over PN " << yval << endl;

    if ( meAmplPNMap ) meAmplPNMap->Fill(xie, xip, yval);

  }

}

