/*
 * \file EBCosmicTask.cc
 * 
 * $Date: 2005/10/07 08:47:46 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBCosmicTask.h>

EBCosmicTask::EBCosmicTask(const edm::ParameterSet& ps, TFile* rootFile){

  logFile.open("EBCosmicTask.log");

  ievt = 0;

  Char_t histo[20];

  rootFile->cd();
  TDirectory* subdir = gDirectory->mkdir("EBCosmicTask");
  subdir->cd();
  subdir = gDirectory->mkdir("Cut");
  subdir = gDirectory->mkdir("Sel");

  rootFile->cd("EBCosmicTask/Cut");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBCT amplitude (cut) SM%02d", i+1);
    hCutMap[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096, "s");
  }

  rootFile->cd("EBCosmicTask/Sel");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBCT amplitude (sel) SM%02d", i+1);
    hSelMap[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096, "s");
  }

}

EBCosmicTask::~EBCosmicTask(){

  logFile.close();

  cout << "EBCosmicTask: analyzed " << ievt << " events" << endl;
}

void EBCosmicTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EBCosmicTask: event " << ievt << " collection size " << neb << endl;

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

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
      }

      float xval = adc * gain;

      float xrms = 1.0;

      if ( xval >= 3.0 * xrms && xval >= xvalmax ) xvalmax = xval;

    }

    hCutMap[ism-1]->Fill(xip, xie, xvalmax);

    hSelMap[ism-1]->Fill(xip, xie, xvalmax);

  }

}

