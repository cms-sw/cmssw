/*
 * \file EcalBarrelMonitorModule.cc
 *
 * $Date: 2006/02/22 06:50:05 $
 * $Revision: 1.81 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const ParameterSet& ps){

  runType_ = -1;

  // this should come from the EcalBarrel run header
  string s = ps.getUntrackedParameter<string>("runType", "unknown");

  if ( s == "cosmic" ) {
    runType_ = COSMIC;
  } else if ( s == "laser" ) {
    runType_ = LASER_STD;
  } else if ( s == "pedestal" ) {
    runType_ = PEDESTAL_STD;
  } else if ( s == "testpulse" ) {
    runType_ = TESTPULSE_MGPA;
  } else if ( s == "electron" ) {
    runType_ = BEAMH4;
  } else if ( s == "electron2" ) {
    runType_ = BEAMH2;
  }

  LogInfo("EcalBarrelMonitor") << " Processing run type: " << runType_ << " (" << s << ")";

  irun_ = ps.getUntrackedParameter<int>("runNumber", 999999);

  LogInfo("EcalBarrelMonitor") << " Processing run: " << irun_;

  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    LogInfo("EcalBarrelMonitor") << " Ecal Barrel Monitoring histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("EcalBarrelMonitor") << " Ecal Barrel Monitoring histograms will NOT be saved";
  }

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    LogInfo("EcalBarrelMonitor") << " verbose switch is ON";
  } else {
    LogInfo("EcalBarrelMonitor") << " verbose switch is OFF";
  }

  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  if ( enableMonitorDaemon_ ) {
    LogInfo("EcalBarrelMonitor") << " enableMonitorDaemon switch is ON";
    Service<MonitorDaemon> daemon;
    daemon.operator->();
  } else {
    LogInfo("EcalBarrelMonitor") << " enableMonitorDaemon switch is OFF";
  }

  // EventDisplay switch
  enableEventDisplay_ = ps.getUntrackedParameter<bool>("enableEventDisplay", false);

  meStatus_ = 0;
  meRun_ = 0;
  meEvt_ = 0;
  meEvtType_ = 0;
  meRunType_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel");
    meStatus_ = dbe_->bookInt("STATUS");

    meRun_ = dbe_->bookInt("RUN");
    meEvt_ = dbe_->bookInt("EVT");

    meEvtType_ = dbe_->book1D("EVTTYPE", "EVTTYPE", 30, 0., 30.);
    meRunType_ = dbe_->bookInt("RUNTYPE");
  }

  if ( meStatus_ ) meStatus_->Fill(-1);

  if ( meRun_ ) meRun_->Fill(-1);
  if ( meEvt_ ) meEvt_->Fill(-1);

  if ( meRunType_ ) meRunType_->Fill(-1);

  // this should give enough time to the control MEs to reach the Collector,
  // and then hopefully the Client

  if ( enableMonitorDaemon_ ) sleep(10);

  meEBDCC_ = 0;

  meEBdigi_ = 0;
  meEBhits_ = 0;

  for (int i = 0; i < 36; i++) {
    meEvent_[i] = 0;
    meOccupancy_[i] = 0;
  }

  Char_t histo[20];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel");

    meEBDCC_ = dbe_->book1D("EBMM SM", "EBMM SM", 36, 1, 37.);

    meEBdigi_ = dbe_->book1D("EBMM digi", "EBMM digi", 100, 0., 61201.);
    meEBhits_ = dbe_->book1D("EBMM hits", "EBMM hits", 100, 0., 61201.);

    if ( enableEventDisplay_ ) {
      dbe_->setCurrentFolder("EcalBarrel/EcalEvent");
      for (int i = 0; i < 36; i++) {
        sprintf(histo, "EBMM event SM%02d", i+1);
        meEvent_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
        if ( meEvent_[i] ) meEvent_[i]->setResetMe(true);
      }
    }

    dbe_->setCurrentFolder("EcalBarrel/EcalOccupancy");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBMM occupancy SM%02d", i+1);
      meOccupancy_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }
  }

  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

}

void EcalBarrelMonitorModule::beginJob(const EventSetup& c){

  ievt_ = 0;

  // begin-of-run
  if ( meStatus_ ) meStatus_->Fill(0);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

}

void EcalBarrelMonitorModule::endJob(void) {

  LogInfo("EcalBarrelMonitor") << "analyzed " << ievt_ << " events";

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

  // this should give enough time to meStatus_ to reach the Collector,
  // and then hopefully the Client, and to allow the Client to complete

  if ( enableMonitorDaemon_ ) sleep(60);

}

void EcalBarrelMonitorModule::analyze(const Event& e, const EventSetup& c){

  // this should give enough time to all the MEs to reach the Collector,
  // and then hopefully the Client, especially when using CollateMEs,
  // even for short runs

  if ( enableMonitorDaemon_ && ievt_ == 0 ) sleep(120);

  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  int nebc = dcchs->size();
  LogInfo("EcalBarrelMonitor") << "event " << ievt_ << " DCC headers collection size " << nebc;

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    dccMap[dcch.id()] = dcch;

    meEBDCC_->Fill((dcch.id()+1)+0.5);

    if ( dccMap[dcch.id()].getRunNumber() != 0 ) irun_ = dccMap[dcch.id()].getRunNumber();

    if ( dccMap[dcch.id()].getRunType() != -1 ) evtType_ = dccMap[dcch.id()].getRunType();

    // uncomment the following line to mix fake 'laser' events w/ cosmic & beam events
//    if ( ievt_ % 10 == 0 && ( runType_ == COSMIC || runType_ == BEAMH4 ) ) evtType_ = LASER_STD;

    if ( evtType_ < 0 || evtType_ > 9 ) {
      LogWarning("EcalBarrelMonitor") << "Unknown event type = " << evtType_;
      evtType_ = -1;
    }

    if ( meEvtType_ ) {

      meEvtType_->Fill(evtType_+0.5);

    }

  }

  ievt_++;

  // running
  if ( meStatus_ ) meStatus_->Fill(1);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

  Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogInfo("EcalBarrelMonitor") << "event " << ievt_ << " digi collection size " << nebd;

  if ( meEBdigi_ ) meEBdigi_->Fill(float(nebd));

  // pause the shipping of monitoring elements
  dbe_->lock();

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    LogDebug("EcalBarrelMonitor") << " det id = " << id;
    LogDebug("EcalBarrelMonitor") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      LogWarning("EcalBarrelMonitor") << " det id = " << id;
      LogWarning("EcalBarrelMonitor") << " sm, eta, phi " << ism << " " << ie << " " << ip;
      LogWarning("EcalBarrelMonitor") << " xie, xip " << xie << " " << xip;
      return;
    }

    if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xie, xip);

  }

  // resume the shipping of monitoring elements
  dbe_->unlock();

  Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalEBUncalibRecHits", hits);

  int nebh = hits->size();
  LogInfo("EcalBarrelMonitor") << "event " << ievt_ << " hits collection size " << nebh;

  if ( meEBhits_ ) meEBhits_->Fill(float(nebh));

  if ( enableEventDisplay_ ) {

    // pause the shipping of monitoring elements
    dbe_->lock();

    for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalUncalibratedRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ie = id.ieta();
      int ip = id.iphi();

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      int ism = id.ism();

      LogDebug("EcalBarrelMonitor") << " det id = " << id;
      LogDebug("EcalBarrelMonitor") << " sm, eta, phi " << ism << " " << ie << " " << ip;

      if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
        LogWarning("EcalBarrelMonitor") << " det id = " << id;
        LogWarning("EcalBarrelMonitor") << " sm, eta, phi " << ism << " " << ie << " " << ip;
        LogWarning("EcalBarrelMonitor") << " xie, xip " << xie << " " << xip;
        return;
      }

      float xval = hit.amplitude();

      LogDebug("EcalBarrelMonitor") << " hit amplitude " << xval;

      if ( xval >= 10 ) {
         if ( meEvent_[ism-1] ) meEvent_[ism-1]->Fill(xie, xip, xval);
      }

    }

    // resume the shipping of monitoring elements
    dbe_->unlock();

  }

//  if ( enableMonitorDaemon_ ) sleep(1);

}

