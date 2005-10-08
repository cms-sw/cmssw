/*
 * \file EBLaserTask.cc
 * 
 * $Date: 2005/10/08 08:55:06 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>

EBLaserTask::EBLaserTask(const edm::ParameterSet& ps, TFile* rootFile){

  logFile.open("EBLaserTask.log");

  ievt = 0;

  Char_t histo[20];

  rootFile->cd();
  TDirectory* subdir = gDirectory->mkdir("EBLaserTask");
  subdir->cd();
  subdir = gDirectory->mkdir("Laser1");
  subdir = gDirectory->mkdir("Laser2");

  rootFile->cd("EBLaserTask/Laser1");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBLT shape SM%02d L1", i+1);
    hShapeMapL1[i] = new TProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 0., 4096., "s");
    sprintf(histo, "EBLT amplitude SM%02d L1", i+1);
    hAmplMapL1[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

  rootFile->cd("EBLaserTask/Laser2");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBLT shape SM%02d L2", i+1);
    hShapeMapL2[i] = new TProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 0., 4096., "s");
    sprintf(histo, "EBLT amplitude SM%02d L2", i+1);
    hAmplMapL2[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

}

EBLaserTask::~EBLaserTask(){

  logFile.close();

  cout << "EBLaserTask: analyzed " << ievt << " events" << endl;
}

void EBLaserTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EBLaserTask: event " << ievt << " collection size " << neb << endl;

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

      int il = 1;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
      }

      if ( il == 1 ) {
          hShapeMap = hShapeMapL1[ism-1];
          hAmplMap = hAmplMapL1[ism-1];
      }
      if ( il == 2 ) {
        hShapeMap = hShapeMapL2[ism-1];
        hAmplMap = hAmplMapL2[ism-1];
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

