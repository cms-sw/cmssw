/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/10/08 08:55:06 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  ievt = 0;

  Char_t histo[20];

  string filename = ps.getUntrackedParameter<string>("fileName");

  rootFile = new TFile(filename.c_str(), "recreate");

  rootFile->cd();

  hEbarrel = new TH1F("EBMM hits", "EBMM hits ", 100, 0., 61200.001);

  TDirectory* subdir = gDirectory->mkdir("EBMonitorEvent");
  subdir->cd();

  rootFile->cd("EBMonitorEvent");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBMM event SM%02d", i+1);
    hEvent[i] = new TH2F(histo, histo, 20, 0., 20., 85, 0., 85.);
  }

  pedestal_task = new EBPedestalTask(ps, rootFile);

  testpulse_task = new EBTestPulseTask(ps, rootFile);

  laser_task = new EBLaserTask(ps, rootFile);

  cosmic_task = new EBCosmicTask(ps, rootFile);

  html_task = new EBHtmlTask(ps, rootFile);

  logFile.open("EcalBarrelMonitorModule.log");

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

  delete pedestal_task;
  delete testpulse_task;
  delete laser_task;
  delete cosmic_task;
  delete html_task;

  rootFile->Write();

  rootFile->Close();

  delete rootFile;

  logFile.close();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt << " events" << endl;

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt << " collection size " << neb << endl;

  hEbarrel->Fill(float(neb));

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

//    logFile << " Dump the ADC counts for this event " << endl;
//    for ( int i=0; i< (*digiItr).size() ; ++i ) {
//      logFile <<  (*digiItr).sample(i) << " ";
//    }       
//    logFile << " " << endl;

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

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      logFile << "ERROR:" << xie << " " << xip << " " << ie << " " << ip << " " << iz << endl;
      return;
    }

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

    hEvent[ism-1]->Fill(xip, xie, xvalmax);

  }

  pedestal_task->analyze(e, c);

  testpulse_task->analyze(e, c);

  laser_task->analyze(e, c);

  cosmic_task->analyze(e, c);

  html_task->analyze(e, c);

}

