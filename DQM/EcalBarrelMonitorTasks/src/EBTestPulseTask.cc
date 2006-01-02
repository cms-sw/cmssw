/*
 * \file EBTestPulseTask.cc
 *
 * $Date: 2005/12/30 10:24:29 $
 * $Revision: 1.32 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>

EBTestPulseTask::EBTestPulseTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBTestPulseTask.log");

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
  }

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask");

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G01", i+1);
      meShapeMapG01_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude SM%02d G01", i+1);
      meAmplMapG01_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude error SM%02d G01", i+1);
      meAmplErrorMapG01_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain06");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G06", i+1);
      meShapeMapG06_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude SM%02d G06", i+1);
      meAmplMapG06_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude error SM%02d G06", i+1);
      meAmplErrorMapG06_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBTestPulseTask/Gain12");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBTPT shape SM%02d G12", i+1);
      meShapeMapG12_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude SM%02d G12", i+1);
      meAmplMapG12_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.);
      sprintf(histo, "EBTPT amplitude error SM%02d G12", i+1);
      meAmplErrorMapG12_[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
   }
  }

  // amplitudeThreshold_ = 200;
  amplitudeThreshold_ = 0;

}

EBTestPulseTask::~EBTestPulseTask(){

//  logFile_.close();

}

void EBTestPulseTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBTestPulseTask::endJob(){

  cout << "EBTestPulseTask: analyzed " << ievt_ << " events" << endl;

}

void EBTestPulseTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  edm::Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

//  int nebd = digis->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " digi collection size " << nebd << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

//    int ie = id.ieta();
//    int ip = id.iphi();

//    float xie = ie - 0.5;
//    float xip = ip - 0.5;

    int ism = id.ism();

    int ic = id.ic();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* meShapeMap = 0;

      if ( sample.gainId() == 1 ) gain = 1./12.;
      if ( sample.gainId() == 2 ) gain = 1./ 6.;
      if ( sample.gainId() == 3 ) gain = 1./ 1.;

      if ( ievt_ >=   1 && ievt_ <=  50 ) meShapeMap = meShapeMapG01_[ism-1];
      if ( ievt_ >=  51 && ievt_ <= 100 ) meShapeMap = meShapeMapG06_[ism-1];
      if ( ievt_ >= 101 && ievt_ <= 150 ) meShapeMap = meShapeMapG12_[ism-1];

      float xval = float(adc) * gain;

      if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

    }

  }

  edm::Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

//  int neh = hits->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " hits collection size " << neb << endl;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    MonitorElement* meAmplMap = 0;

    if ( ievt_ >=   1 && ievt_ <=  50 ) meAmplMap = meAmplMapG01_[ism-1];
    if ( ievt_ >=  50 && ievt_ <= 100 ) meAmplMap = meAmplMapG06_[ism-1];
    if ( ievt_ >= 100 && ievt_ <= 150 ) meAmplMap = meAmplMapG12_[ism-1];

    float xval = hit.amplitude();

//    logFile_ << " hit amplitude " << xval << endl;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

    MonitorElement* meAmplErrorMap = 0;

    if ( ievt_ >=   1 && ievt_ <=  50 ) meAmplErrorMap = meAmplErrorMapG01_[ism-1];
    if ( ievt_ >=  50 && ievt_ <= 100 ) meAmplErrorMap = meAmplErrorMapG06_[ism-1];
    if ( ievt_ >= 100 && ievt_ <= 150 ) meAmplErrorMap = meAmplErrorMapG12_[ism-1];

    // cout << "Crystal " << ie << " " << ip << " Amplitude = " << xval << endl;

    if ( xval < amplitudeThreshold_ ) {

      if ( meAmplErrorMap ) meAmplErrorMap->Fill(xie, xip);

    }

  }

}

