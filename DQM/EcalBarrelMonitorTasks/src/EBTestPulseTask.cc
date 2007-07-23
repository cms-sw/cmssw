/*
 * \file EBTestPulseTask.cc
 *
 * $Date: 2007/02/17 12:33:37 $
 * $Revision: 1.61 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTestPulseTask::EBTestPulseTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    meShapeMapG01_[i] = 0;
    meAmplMapG01_[i] = 0;
    meAmplErrorMapG01_[i] = 0;
    meShapeMapG06_[i] = 0;
    meAmplMapG06_[i] = 0;
    meAmplErrorMapG06_[i] = 0;
    meShapeMapG12_[i] = 0;
    meAmplMapG12_[i] = 0;
    meAmplErrorMapG12_[i] = 0;
    mePnAmplMapG01_[i] = 0;
    mePnPedMapG01_[i] = 0;
    mePnAmplMapG16_[i] = 0;
    mePnPedMapG16_[i] = 0;
  }

  // amplitudeThreshold_ = 200;
  amplitudeThreshold_ = 0;

}

EBTestPulseTask::~EBTestPulseTask(){

}

void EBTestPulseTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask");
    dbe->rmdir("EcalBarrel/EBTestPulseTask");
  }

}

void EBTestPulseTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask");

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G01", i+1);
      meShapeMapG01_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapG01_[i], i+1);
      sprintf(histo, "EBTPT amplitude SM%02d G01", i+1);
      meAmplMapG01_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapG01_[i], i+1);
      sprintf(histo, "EBTPT amplitude error SM%02d G01", i+1);
      meAmplErrorMapG01_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe->tag(meAmplErrorMapG01_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain06");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G06", i+1);
      meShapeMapG06_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapG06_[i], i+1);
      sprintf(histo, "EBTPT amplitude SM%02d G06", i+1);
      meAmplMapG06_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapG06_[i], i+1);
      sprintf(histo, "EBTPT amplitude error SM%02d G06", i+1);
      meAmplErrorMapG06_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe->tag(meAmplErrorMapG06_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G12", i+1);
      meShapeMapG12_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapG12_[i], i+1);
      sprintf(histo, "EBTPT amplitude SM%02d G12", i+1);
      meAmplMapG12_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapG12_[i], i+1);
      sprintf(histo, "EBTPT amplitude error SM%02d G12", i+1);
      meAmplErrorMapG12_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe->tag(meAmplErrorMapG12_[i], i+1);
   }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01", i+1);
      mePnAmplMapG01_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG01_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01", i+1);
      mePnPedMapG01_[i] =  dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG01_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16", i+1);
      mePnAmplMapG16_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG16_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16", i+1);
      mePnPedMapG16_[i] =  dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG16_[i], i+1);
    }

  }

}

void EBTestPulseTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask");

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapG01_[i] ) dbe->removeElement( meShapeMapG01_[i]->getName() );
      meShapeMapG01_[i] = 0;
      if ( meAmplMapG01_[i] ) dbe->removeElement( meAmplMapG01_[i]->getName() );
      meAmplMapG01_[i] = 0;
      if ( meAmplErrorMapG01_[i] ) dbe->removeElement( meAmplErrorMapG01_[i]->getName() );
      meAmplErrorMapG01_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain06");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapG06_[i] ) dbe->removeElement( meShapeMapG06_[i]->getName() );
      meShapeMapG06_[i] = 0;
      if ( meAmplMapG06_[i] ) dbe->removeElement( meAmplMapG06_[i]->getName() );
      meAmplMapG06_[i] = 0;
      if ( meAmplErrorMapG06_[i] ) dbe->removeElement( meAmplErrorMapG06_[i]->getName() );
      meAmplErrorMapG06_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapG12_[i] ) dbe->removeElement( meShapeMapG12_[i]->getName() );
      meShapeMapG12_[i] = 0;
      if ( meAmplMapG12_[i] ) dbe->removeElement( meAmplMapG12_[i]->getName() );
      meAmplMapG12_[i] = 0;
      if ( meAmplErrorMapG12_[i] ) dbe->removeElement( meAmplErrorMapG12_[i]->getName() );
      meAmplErrorMapG12_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG01_[i] ) dbe->removeElement( mePnAmplMapG01_[i]->getName() );
      mePnAmplMapG01_[i] = 0;
      if ( mePnPedMapG01_[i] ) dbe->removeElement( mePnPedMapG01_[i]->getName() );
      mePnPedMapG01_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Gain16");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG16_[i] ) dbe->removeElement( mePnAmplMapG16_[i]->getName() );
      mePnAmplMapG16_[i] = 0;
      if ( mePnPedMapG16_[i] ) dbe->removeElement( mePnPedMapG16_[i]->getName() );
      mePnPedMapG16_[i] = 0;
    }

  }

  init_ = false;

}

void EBTestPulseTask::endJob(void){

  LogInfo("EBTestPulseTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBTestPulseTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(dcch.id());
    if ( i != dccMap.end() ) continue;

    dccMap[dcch.id()] = dcch;

    if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ) enable = true;

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogDebug("EBTestPulseTask") << "event " << ievt_ << " digi collection size " << nebd;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ) continue;

    LogDebug("EBTestPulseTask") << " det id = " << id;
    LogDebug("EBTestPulseTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* meShapeMap = 0;

      if ( sample.gainId() == 1 ) gain = 1./12.;
      if ( sample.gainId() == 2 ) gain = 1./ 6.;
      if ( sample.gainId() == 3 ) gain = 1./ 1.;

      if ( dccMap[ism-1].getMgpaGain() == 3 ) meShapeMap = meShapeMapG01_[ism-1];
      if ( dccMap[ism-1].getMgpaGain() == 2 ) meShapeMap = meShapeMapG06_[ism-1];
      if ( dccMap[ism-1].getMgpaGain() == 1 ) meShapeMap = meShapeMapG12_[ism-1];

//      float xval = float(adc) * gain;
      float xval = float(adc);

      if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

    }

  }

  Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalUncalibRecHitsEB", hits);

  int neh = hits->size();
  LogDebug("EBTestPulseTask") << "event " << ievt_ << " hits collection size " << neh;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ) continue;

    LogDebug("EBTestPulseTask") << " det id = " << id;
    LogDebug("EBTestPulseTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    MonitorElement* meAmplMap = 0;

    if ( dccMap[ism-1].getMgpaGain() == 3 ) meAmplMap = meAmplMapG01_[ism-1];
    if ( dccMap[ism-1].getMgpaGain() == 2 ) meAmplMap = meAmplMapG06_[ism-1];
    if ( dccMap[ism-1].getMgpaGain() == 1 ) meAmplMap = meAmplMapG12_[ism-1];

    float xval = hit.amplitude();
    if ( xval <= 0. ) xval = 0.0;

//    if ( dccMap[ism-1].getMgpaGain() == 3 ) xval = xval * 1./12.;
//    if ( dccMap[ism-1].getMgpaGain() == 2 ) xval = xval * 1./ 2.;
//    if ( dccMap[ism-1].getMgpaGain() == 1 ) xval = xval * 1./ 1.;

    LogDebug("EBTestPulseTask") << " hit amplitude " << xval;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

    MonitorElement* meAmplErrorMap = 0;

    if ( dccMap[ism-1].getMgpaGain() == 3 ) meAmplErrorMap = meAmplErrorMapG01_[ism-1];
    if ( dccMap[ism-1].getMgpaGain() == 2 ) meAmplErrorMap = meAmplErrorMapG06_[ism-1];
    if ( dccMap[ism-1].getMgpaGain() == 1 ) meAmplErrorMap = meAmplErrorMapG12_[ism-1];

    LogDebug("EBTestPulseTask") << "Crystal " << ie << " " << ip << " Amplitude = " << xval;

    if ( xval < amplitudeThreshold_ ) {

      if ( meAmplErrorMap ) meAmplErrorMap->Fill(xie, xip);

    }

  }

  Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

  int nep = pns->size();
  LogDebug("EBTestPulseTask") << "event " << ievt_ << " pns collection size " << nep;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();

//    int ism = id.ism();
    int ism = id.iDCCId();

    int num = id.iPnId();

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::TESTPULSE_MGPA ) continue;

    LogDebug("EBTestPulseTask") << " det id = " << id;
    LogDebug("EBTestPulseTask") << " sm, num " << ism << " " << num;

    float xvalped = 0.;

    for (int i = 0; i < 4; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      MonitorElement* mePNPed = 0;

      if ( sample.gainId() == 0 ) mePNPed = mePnPedMapG01_[ism-1];
      if ( sample.gainId() == 1 ) mePNPed = mePnPedMapG16_[ism-1];

      float xval = float(adc);

      if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

      xvalped = xvalped + xval;

    }

    xvalped = xvalped / 4;

    float xvalmax = 0.;

    MonitorElement* mePN = 0;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      float xval = float(adc);

      if ( xval >= xvalmax ) xvalmax = xval;

    }

    xvalmax = xvalmax - xvalped;

    if ( pn.sample(0).gainId() == 0 ) mePN = mePnAmplMapG01_[ism-1];
    if ( pn.sample(0).gainId() == 1 ) mePN = mePnAmplMapG16_[ism-1];

    if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

  }

}

