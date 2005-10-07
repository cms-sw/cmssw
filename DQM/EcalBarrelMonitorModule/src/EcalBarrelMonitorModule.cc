/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/10/07 10:06:32 $
 * $Revision: 1.3 $
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  ievt = 0;

  string filename = ps.getUntrackedParameter<string>("fileName");

  rootFile = new TFile(filename.c_str(), "recreate");

  rootFile->cd();

  hEbarrel = new TH1F("EBMM1", "EB hits ", 100, 0., 61200.001);

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

  }

  pedestal_task->analyze(e, c);

  testpulse_task->analyze(e, c);

  laser_task->analyze(e, c);

  cosmic_task->analyze(e, c);

  html_task->analyze(e, c);

}

