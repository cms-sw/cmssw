/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/12/12 07:26:26 $
 * $Revision: 1.63 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

//  logFile_.open("EcalBarrelMonitorModule.log");

  string s = ps.getUntrackedParameter<string>("runType", "unknown");

  if ( s == "cosmic" ) {
    runType_ = 0;
  } else if ( s == "laser" ) {
    runType_ = 1;
  } else if ( s == "pedestal" ) {
    runType_ = 2;
  } else if ( s == "testpulse" ) {
    runType_ = 3;
  }

  irun_ = ps.getUntrackedParameter<int>("runNumber", 999999);

  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    cout << " Ecal Barrel Monitoring histograms will be saved to '" << outputFile_.c_str() << "'" << endl;
  } else {
    cout << " Ecal Barrel Monitoring histograms will NOT be saved" << endl;
  }

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    cout << " verbose switch is ON" << endl;
  } else {
    cout << " verbose switch is OFF" << endl;
  }

  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }

  //We put this here for the moment since there is no better place 
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  Char_t histo[20];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel");
    meStatus_  = dbe_->bookInt("STATUS");
    meStatus_->setResetMe(true);
    meRun_     = dbe_->bookInt("RUN");
    meRun_->setResetMe(true);
    meEvt_     = dbe_->bookInt("EVT");
    meEvt_->setResetMe(true);

    meEvtType_ = dbe_->book1D("EVTTYPE", "EVTTYPE", 10, 0., 10.);
    meRunType_ = dbe_->bookInt("RUNTYPE");
    meRunType_->setResetMe(true);

    // this should give enough time to the control MEs to reach the Collector,
    // and then hopefully the clients
    sleep(10);

    dbe_->setCurrentFolder("EcalBarrel");
    meEBdigi_ = dbe_->book1D("EBMM digi", "EBMM digi", 100, 0., 61201.);
    meEBhits_ = dbe_->book1D("EBMM hits", "EBMM hits", 100, 0., 61201.);

    dbe_->setCurrentFolder("EcalBarrel/EBMonitorEvent");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBMM event SM%02d", i+1);
      meEvent_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meEvent_[i]->setResetMe(true);
    }
  }

  integrity_task_ = 0;

  cosmic_task_ = 0;
  laser_task_ = 0;
  pndiode_task_ = 0;
  pedestal_task_ = 0;
  pedpresample_task_ = 0;
  testpulse_task_ = 0;

                       integrity_task_ = new EBIntegrityTask(ps, dbe_);

  if ( runType_ == 0 ) cosmic_task_ = new EBCosmicTask(ps, dbe_);

  if ( runType_ == 0 || runType_ == 1 ) {
                       laser_task_ = new EBLaserTask(ps, dbe_);
                       pndiode_task_ = new EBPnDiodeTask(ps, dbe_);
  }

  if ( runType_ == 2 ) pedestal_task_ = new EBPedestalTask(ps, dbe_);
                       pedpresample_task_ = new EBPedPreSampleTask(ps, dbe_);

  if ( runType_ == 3 ) testpulse_task_ = new EBTestPulseTask(ps, dbe_);

  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }

  // this should give enough time to all the MEs to reach the Collector,
  // and then hopefully the clients, even for short runs
  sleep(120);

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

  if ( integrity_task_ ) delete integrity_task_;

  if ( cosmic_task_ ) delete cosmic_task_;
  if ( laser_task_ ) delete laser_task_;
  if ( pndiode_task_ ) delete pndiode_task_;
  if ( pedestal_task_ ) delete pedestal_task_;
  if ( pedpresample_task_ ) delete pedpresample_task_;
  if ( testpulse_task_ ) delete testpulse_task_;

//  logFile_.close();

}

void EcalBarrelMonitorModule::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

  // begin-of-run
  if ( meStatus_ ) meStatus_->Fill(0);

  if ( integrity_task_ ) integrity_task_->beginJob(c);

  if ( cosmic_task_ ) cosmic_task_->beginJob(c);
  if ( laser_task_ ) laser_task_->beginJob(c);
  if ( pndiode_task_ ) pndiode_task_->beginJob(c);
  if ( pedestal_task_ ) pedestal_task_->beginJob(c);
  if ( pedpresample_task_ ) pedpresample_task_->beginJob(c);
  if ( testpulse_task_ ) testpulse_task_->beginJob(c);

}

void EcalBarrelMonitorModule::endJob(void) {

  if ( integrity_task_ ) integrity_task_->endJob();

  if ( cosmic_task_ ) cosmic_task_->endJob();
  if ( laser_task_ ) laser_task_->endJob();
  if ( pndiode_task_ ) pndiode_task_->endJob();
  if ( pedestal_task_ ) pedestal_task_->endJob();
  if ( pedpresample_task_ ) pedpresample_task_->endJob();
  if ( testpulse_task_ ) testpulse_task_->endJob();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt_ << " events" << endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

  // this should give enough time to meStatus_ to reach the Collector,
  // and then hopefully the clients ...
  sleep(60);

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  // running
  if ( meStatus_ ) meStatus_->Fill(1);

  ievt_++;

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  evtType_ = runType_;

  if ( meEvtType_ ) meEvtType_->Fill(evtType_+0.5);
  if ( meRunType_ ) meRunType_->Fill(runType_);

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt_ << " digi collection size " << nebd << endl;

  if ( meEBdigi_ ) meEBdigi_->Fill(float(nebd));

  edm::Handle<EcalUncalibratedRecHitCollection>  hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

  int nebh = hits->size();

  cout << "EcalBarrelMonitorModule: event " << ievt_ << " hits collection size " << nebh << endl;

  if ( meEBhits_ ) meEBhits_->Fill(float(nebh));

  // pause the shipping of monitoring elements
  dbe_->lock();

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      cout << " det id = " << id << endl;
      cout << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;
      cout << "ERROR:" << xie << " " << xip << " " << ie << " " << ip << " " << iz << endl;
      return;
    }

    float xval = hit.amplitude();

//    logFile_ << " hit amplitude " << xval << endl;

    if ( xval >= 10 ) {
       if ( meEvent_[ism-1] ) meEvent_[ism-1]->Fill(xie, xip, xval);
    }

  }

  // resume the shipping of monitoring elements
  dbe_->unlock();

  if (                  integrity_task_ ) integrity_task_->analyze(e, c);

  if ( evtType_ == 0 && cosmic_task_ ) cosmic_task_->analyze(e, c);

  if ( evtType_ == 1 && laser_task_ ) laser_task_->analyze(e, c);

  if ( evtType_ == 1 && pndiode_task_ ) pndiode_task_->analyze(e, c);

  if ( evtType_ == 2 && pedestal_task_ ) pedestal_task_->analyze(e, c);

  if (                  pedpresample_task_ ) pedpresample_task_->analyze(e, c);

  if ( evtType_ == 3 && testpulse_task_ ) testpulse_task_->analyze(e, c);

//  sleep(1);

}

