/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/10/28 10:22:30 $
 * $Revision: 1.24 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  logFile.open("EcalBarrelMonitorModule.log");

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  dbe->setVerbose(1);

  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  Char_t histo[20];

  outputFile = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile.size() != 0 ) {
    cout << "Ecal Barrel Monitoring histograms will be saved to " << outputFile.c_str() << endl;
  }

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");
    meStatus  = dbe->bookInt("STATUS");
    meRun     = dbe->bookInt("RUN");
    meEvt     = dbe->bookInt("EVT");

    dbe->setCurrentFolder("EcalBarrel");
    meEBdigi = dbe->book1D("EBMM digi", "EBMM digi", 100, 0., 61201.);
    meEBhits = dbe->book1D("EBMM hits", "EBMM hits", 100, 0., 61201.);

    dbe->setCurrentFolder("EcalBarrel/EBMonitorEvent");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBMM event SM%02d", i+1);
      meEvent[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meEvent[i]->setResetMe(true);
    }
  }

  pedestal_task  = new EBPedestalTask(ps, dbe);

  testpulse_task = new EBTestPulseTask(ps, dbe);

  laser_task     = new EBLaserTask(ps, dbe);

  cosmic_task    = new EBCosmicTask(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

  delete pedestal_task;

  delete testpulse_task;

  delete laser_task;

  delete cosmic_task;

  logFile.close();

}

void EcalBarrelMonitorModule::beginJob(const edm::EventSetup& c){

  ievt = 0;

  if ( meStatus ) meStatus->Fill(0);

  pedestal_task->beginJob(c);

  testpulse_task->beginJob(c);

  laser_task->beginJob(c);

  cosmic_task->beginJob(c);

}

void EcalBarrelMonitorModule::endJob(void) {

  pedestal_task->endJob();

  testpulse_task->endJob();

  laser_task->endJob();

  cosmic_task->endJob();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt << " events" << endl;

  if ( meStatus ) meStatus->Fill(2);

  if ( outputFile.size() != 0  && dbe ) dbe->save(outputFile);

  sleep(60);
}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( meStatus ) meStatus->Fill(1);

  ievt++;

  if ( meRun ) meRun->Fill(14316);
  if ( meEvt ) meEvt->Fill(ievt);

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt << " digi collection size " << nebh << endl;

  if ( meEBdigi ) meEBdigi->Fill(float(nebd));

  edm::Handle<EcalUncalibratedRecHitCollection>  hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

  int nebh = hits->size();

  cout << "EcalBarrelMonitorModule: event " << ievt << " hits collection size " << nebh << endl;

  if ( meEBhits ) meEBhits->Fill(float(nebh));

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      cout << " det id = " << id << endl;
      cout << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;
      cout << "ERROR:" << xie << " " << xip << " " << ie << " " << ip << " " << iz << endl;
      return;
    }

// average rms per crystal

    float xrms = 1.2;

    float xval = hit.amplitude();

//    logFile << " hit amplitude " << xval << endl;

    if ( xval >= 3.0 * xrms ) {
       if ( meEvent[ism-1] ) meEvent[ism-1]->Fill(xie, xip, xval);
    }

  }

  pedestal_task->analyze(e, c);

  testpulse_task->analyze(e, c);

  laser_task->analyze(e, c);

  cosmic_task->analyze(e, c);

}

