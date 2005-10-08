/*
 * \file EBTestPulseTask.cc
 * 
 * $Date: 2005/10/07 08:47:46 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBTestPulseTask.h>

EBTestPulseTask::EBTestPulseTask(const edm::ParameterSet& ps, TFile* rootFile){

  logFile.open("EBTestPulseTask.log");

  ievt = 0;

  Char_t histo[20];

  rootFile->cd();
  TDirectory* subdir = gDirectory->mkdir("EBTestPulseTask");
  subdir->cd();
  subdir = gDirectory->mkdir("Gain01");
  subdir = gDirectory->mkdir("Gain06");
  subdir = gDirectory->mkdir("Gain12");

  rootFile->cd("EBTestPulseTask/Gain01");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBTT shape SM%02d G01", i+1);
    hShapeMapG01[i] = new TProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 0., 4096., "s");
    sprintf(histo, "EBTT amplitude SM%02d G01", i+1);
    hAmplMapG01[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

  rootFile->cd("EBTestPulseTask/Gain06");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBTT shape SM%02d G06", i+1);
    hShapeMapG06[i] = new TProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 0., 4096., "s");
    sprintf(histo, "EBTT amplitude SM%02d G06", i+1);
    hAmplMapG06[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

  rootFile->cd("EBTestPulseTask/Gain12");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBTT shape SM%02d G12", i+1);
    hShapeMapG12[i] = new TProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 0., 4096., "s");
    sprintf(histo, "EBTT amplitude SM%02d G12", i+1);
    hAmplMapG12[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

}

EBTestPulseTask::~EBTestPulseTask(){

  logFile.close();

  cout << "EBTestPulseTask: analyzed " << ievt << " events" << endl;

}

void EBTestPulseTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EBTestPulseTask: event " << ievt << " collection size " << neb << endl;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = EBMonitorUtils::getSuperModuleID(ip, iz);

    logFile << " det id = " << id << endl;
    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    float xvalmax = 0.;

    TProfile2D* hAmplMap = 0;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      TProfile2D* hShapeMap = 0;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
        hShapeMap = hShapeMapG12[ism-1];
        hAmplMap = hAmplMapG12[ism-1];
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
        hShapeMap = hShapeMapG06[ism-1];
        hAmplMap = hAmplMapG06[ism-1];
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
        hShapeMap = hShapeMapG01[ism-1];
        hAmplMap = hAmplMapG01[ism-1];
      }

      float xval = adc * gain;

      int ic = EBMonitorUtils::getCrystalID(ie, ip);

      if ( hShapeMap ) hShapeMap->Fill( ic - 0.5, i + 0.5, xval);

      float xrms = 1.0;

      if ( xval >= 3.0 * xrms && xval >= xvalmax ) xvalmax = xval;

    }

    if ( hAmplMap ) hAmplMap->Fill(xip, xie, xvalmax);

  }

}

