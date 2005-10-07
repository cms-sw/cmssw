/*
 * \file EBPedestalTask.cc
 * 
 * $Date: 2005/08/05 14:34:03 $
 * $Revision: 1.2 $
 *
*/

#include <EcalMonitor/EBMonitorTasks/interface/EBPedestalTask.h>

EBPedestalTask::EBPedestalTask(const edm::ParameterSet& ps, TFile* rootFile){

  logFile.open("EBPedestalTask.log");

  ievt = 0;

  Char_t histo[20];

  rootFile->cd();
  TDirectory* subdir = gDirectory->mkdir("EBPedestalTask");
  subdir->cd();
  subdir = gDirectory->mkdir("Gain01");
  subdir = gDirectory->mkdir("Gain06");
  subdir = gDirectory->mkdir("Gain12");

  rootFile->cd("EBPedestalTask/Gain01");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G01", i+1);
    hPedMapG01[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

  rootFile->cd("EBPedestalTask/Gain06");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G06", i+1);
    hPedMapG06[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

  rootFile->cd("EBPedestalTask/Gain12");
  for (int i = 0; i < 36 ; i++) {
    sprintf(histo, "EBPT pedestal SM%02d G12", i+1);
    hPedMapG12[i] = new TProfile2D(histo, histo, 20, 0., 20., 85, 0., 85., 0., 4096., "s");
  }

}

EBPedestalTask::~EBPedestalTask(){

  logFile.close();

  cout << "EBPedestalTask: analyzed " << ievt << " events" << endl;
}

void EBPedestalTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EBPedestalTask: event " << ievt << " collection size " << neb << endl;

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

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      TProfile2D* hPedMap = 0;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
        hPedMap = hPedMapG12[ism-1];
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
        hPedMap = hPedMapG06[ism-1];
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
        hPedMap = hPedMapG01[ism-1];
      }

      float xval = adc * gain;

      if ( i <= 3 ) {
        if ( hPedMap ) hPedMap->Fill(xip, xie, xval);
      }
      if ( i >= 4 ) {
        if ( hPedMap ) hPedMap->Fill(xip, xie, xval);
      }
    }

  }

}

