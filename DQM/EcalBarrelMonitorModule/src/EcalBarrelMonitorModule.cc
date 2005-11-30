/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/11/24 18:19:18 $
 * $Revision: 1.54 $
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

  dbe_ = EcalBarrelMonitorDaemon::dbe();

  dbe_->setVerbose(1);

  Char_t histo[20];

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "Ecal Barrel Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

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

  cosmic_task_ = new EBCosmicTask(ps, dbe_);
  laser_task_ = new EBLaserTask(ps, dbe_);
  pndiode_task_ = new EBPnDiodeTask(ps, dbe_);
  pedestal_task_ = new EBPedestalTask(ps, dbe_);
  pedpresample_task_ = new EBPedPreSampleTask(ps, dbe_);
  testpulse_task_ = new EBTestPulseTask(ps, dbe_);

  if ( dbe_ ) dbe_->showDirStructure();

  // this should give enough time to the ME to reach the Collector,
  // and then hopefully the clients, even for short runs
  sleep(60);

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

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

  if ( meStatus_ ) meStatus_->Fill(0);

  cosmic_task_->beginJob(c);
  laser_task_->beginJob(c);
  pndiode_task_->beginJob(c);
  pedestal_task_->beginJob(c);
  pedpresample_task_->beginJob(c);
  testpulse_task_->beginJob(c);

}

void EcalBarrelMonitorModule::endJob(void) {

  cosmic_task_->endJob();
  laser_task_->endJob();
  pndiode_task_->endJob();
  pedestal_task_->endJob();
  pedpresample_task_->endJob();
  testpulse_task_->endJob();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt_ << " events" << endl;

  if ( meStatus_ ) meStatus_->Fill(2);

  if ( outputFile_.size() != 0  && dbe_ ) dbe_->save(outputFile_);

  // this should give enough time to meStatus_ to reach the Collector,
  // and then hopefully the clients ...
  sleep(60);

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

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

  if ( evtType_ == 0 ) cosmic_task_->analyze(e, c);

  if ( evtType_ == 1 ) laser_task_->analyze(e, c);

  if ( evtType_ == 1 ) pndiode_task_->analyze(e, c);

  if ( evtType_ == 2 ) pedestal_task_->analyze(e, c);

                       pedpresample_task_->analyze(e, c);

  if ( evtType_ == 3 ) testpulse_task_->analyze(e, c);

//  sleep(1);

}

