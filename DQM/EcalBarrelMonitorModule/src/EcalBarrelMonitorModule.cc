/*
 * \file EcalBarrelMonitorModule.cc
 *
 * $Date: 2006/01/29 17:21:26 $
 * $Revision: 1.77 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

//  logFile_.open("EcalBarrelMonitorModule.log");

  // this is a hack, used to fake the EcalBarrel run & event headers
  TH1F* tmp = new TH1F("tmp", "tmp", 2, 0., 1.);
  tmp->SetBinContent(1, -1.);
  tmp->SetBinContent(2, -1.);

  runType_ = -1;

  // this should come from the EcalBarrel run header
  string s = ps.getUntrackedParameter<string>("runType", "unknown");

  if ( s == "cosmic" ) {
    runType_ = 0;
  } else if ( s == "laser" ) {
    runType_ = 1;
  } else if ( s == "pedestal" ) {
    runType_ = 2;
  } else if ( s == "testpulse" ) {
    runType_ = 3;
  } else if ( s == "electron" ) {
    runType_ = 4;
  }

  // this is a hack, used to fake the EcalBarrel run header
  tmp->SetBinContent(1, runType_);

  cout << " Processing run type: " << runType_ << " (" << s << ")" << endl;

  irun_ = ps.getUntrackedParameter<int>("runNumber", 999999);

  cout << " Processing run: " << irun_ << endl;

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

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  if ( enableMonitorDaemon_ ) {
    cout << " enableMonitorDaemon switch is ON" << endl;
    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
  } else {
    cout << " enableMonitorDaemon switch is OFF" << endl;
  }

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

    meEvtType_ = dbe_->book1D("EVTTYPE", "EVTTYPE", 10, 0., 10.);
    meRunType_ = dbe_->bookInt("RUNTYPE");
  }

  if ( meStatus_ ) meStatus_->Fill(-1);

  if ( meRun_ ) meRun_->Fill(-1);
  if ( meEvt_ ) meEvt_->Fill(-1);

  if ( meRunType_ ) meRunType_->Fill(-1);

  // this should give enough time to the control MEs to reach the Collector,
  // and then hopefully the clients
  if ( enableMonitorDaemon_ ) sleep(10);

  meEBdigi_ = 0;
  meEBhits_ = 0;

  for (int i = 0; i < 36 ; i++) {
    meEvent_[i] = 0;
    meOccupancy_[i] = 0;
  }

  Char_t histo[20];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel");
    meEBdigi_ = dbe_->book1D("EBMM digi", "EBMM digi", 100, 0., 61201.);
    meEBhits_ = dbe_->book1D("EBMM hits", "EBMM hits", 100, 0., 61201.);

    dbe_->setCurrentFolder("EcalBarrel/EcalEvent");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBMM event SM%02d", i+1);
      meEvent_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      if ( meEvent_[i] ) meEvent_[i]->setResetMe(true);
    }

    dbe_->setCurrentFolder("EcalBarrel/EcalOccupancy");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBMM occupancy SM%02d", i+1);
      meOccupancy_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }
  }

  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

//  logFile_.close();

}

void EcalBarrelMonitorModule::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

  // begin-of-run
  if ( meStatus_ ) meStatus_->Fill(0);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

  // this should give enough time to all the MEs to reach the Collector,
  // and then hopefully the clients, even for short runs
  if ( enableMonitorDaemon_ ) sleep(120);

}

void EcalBarrelMonitorModule::endJob(void) {

  cout << "EcalBarrelMonitorModule: analyzed " << ievt_ << " events" << endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

  // this should give enough time to meStatus_ to reach the Collector,
  // and then hopefully the clients ...
  if ( enableMonitorDaemon_ ) sleep(60);

}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  // running
  if ( meStatus_ ) meStatus_->Fill(1);

  if ( meRun_ ) meRun_->Fill(irun_);
  if ( meEvt_ ) meEvt_->Fill(ievt_);

  if ( meRunType_ ) meRunType_->Fill(runType_);

  // this should come from the EcalBarrel event header
  evtType_ = runType_;

  // uncomment the following line to add fake 'laser' events
//  if ( ievt_ % 10 == 0 && ( runType_ == 0 || runType_ == 4 ) ) evtType_ = 1;

  // this is a hack, used to fake the EcalBarrel event header
  TH1F* tmp = (TH1F*) gROOT->FindObjectAny("tmp");
  tmp->SetBinContent(2, evtType_);

  if ( meEvtType_ ) meEvtType_->Fill(evtType_+0.5);

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt_ << " digi collection size " << nebd << endl;

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

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      cout << " det id = " << id << endl;
      cout << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;
      cout << "ERROR:" << xie << " " << xip << endl;
      return;
    }

    if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xie, xip);

  }

  // resume the shipping of monitoring elements
  dbe_->unlock();

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

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

//    logFile_ << " det id = " << id << endl;
//    logFile_ << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      cout << " det id = " << id << endl;
      cout << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;
      cout << "ERROR:" << xie << " " << xip << endl;
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

//  if ( enableMonitorDaemon_ ) sleep(1);

}

