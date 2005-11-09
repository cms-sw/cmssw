/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/11/08 17:52:08 $
 * $Revision: 1.35 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  logFile.open("EcalBarrelMonitorModule.log");

  string s = ps.getUntrackedParameter<string>("runType","unknown");

  if ( s == "cosmic" ) {
    runType = 0;
  } else if ( s == "laser" ) {
    runType = 1;
  } else if ( s == "pedestal" ) {
    runType = 2;
  } else if ( s == "testpulse" ) {
    runType = 3;
  }

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
    meRunType = dbe->bookInt("RUNTYPE");

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

  cosmic_task    = new EBCosmicTask(ps, dbe);

  laser_task     = new EBLaserTask(ps, dbe);

  pedestal_task  = new EBPedestalTask(ps, dbe);

  testpulse_task = new EBTestPulseTask(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

  delete cosmic_task;

  delete laser_task;

  delete pedestal_task;

  delete testpulse_task;

  logFile.close();

}

void EcalBarrelMonitorModule::beginJob(const edm::EventSetup& c){

  ievt = 0;

  if ( meStatus ) meStatus->Fill(0);

  cosmic_task->beginJob(c);

  laser_task->beginJob(c);

  pedestal_task->beginJob(c);

  testpulse_task->beginJob(c);

}

void EcalBarrelMonitorModule::endJob(void) {

  cosmic_task->endJob();

  laser_task->endJob();

  pedestal_task->endJob();

  testpulse_task->endJob();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt << " events" << endl;

  if ( meStatus ) meStatus->Fill(2);

  if ( outputFile.size() != 0  && dbe ) dbe->save(outputFile);

  // this is to give enough time to the meStatus to reach the clients ...
  sleep(60);

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( meStatus ) meStatus->Fill(1);

  ievt++;

  if ( meRun ) meRun->Fill(14316);
  if ( meEvt ) meEvt->Fill(ievt);

  if ( meRunType ) meRunType->Fill(runType);

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt << " digi collection size " << nebd << endl;

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

    float xval = 0.001 * hit.amplitude();

//    logFile << " hit amplitude " << xval << endl;

    if ( xval >= 10 ) {
       if ( meEvent[ism-1] ) meEvent[ism-1]->Fill(xie, xip, xval);
    }

  }

  if ( runType == 0 ) cosmic_task->analyze(e, c);

  if ( runType == 1 ) laser_task->analyze(e, c);

  if ( runType == 2 ) pedestal_task->analyze(e, c);

  if ( runType == 3 ) testpulse_task->analyze(e, c);

//  sleep(1);

}

