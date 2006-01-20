/*
 * \file EcalBarrelMonitorClient.cc
 *
 * $Date: 2006/01/19 14:18:17 $
 * $Revision: 1.81 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h>

EcalBarrelMonitorClient::EcalBarrelMonitorClient(const edm::ParameterSet& ps){

  cout << endl;
  cout << " *** Ecal Barrel Generic Monitor Client ***" << endl;
  cout << endl;

  mui_ = 0;

  integrity_client_      = 0;

  cosmic_client_         = 0;
  laser_client_          = 0;
  pedestal_client_       = 0;
  pedestalonline_client_ = 0;
  testpulse_client_      = 0;
  electron_client_       = 0;

  begin_run_done_ = false;
  end_run_done_   = false;

  forced_begin_run_ = false;
  forced_end_run_   = false;

  h_ = 0;

  status_  = "unknown";
  run_     = -1;
  evt_     = -1;
  runtype_ = "unknown";

  last_jevt_   = -1;
  last_update_ = 0;

  unknowns_ = 0;

  // DQM default client name

  clientName_ = ps.getUntrackedParameter<string>("clientName", "EcalBarrelMonitorClient");

  // DQM default collector host name

  hostName_ = ps.getUntrackedParameter<string>("hostName", "localhost");

  // DQM default host port

  hostPort_ = ps.getUntrackedParameter<int>("hostPort", 9090);;

  cout << " Client '" << clientName_ << "' " << endl
       << " Collector on host '" << hostName_ << "'"
       << " on port '" << hostPort_ << "'" << endl;

  // DQM ROOT output

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");

  // Ecal Cond DB

  dbName_ = ps.getUntrackedParameter<string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<string>("dbHostName", "");
  dbUserName_ = ps.getUntrackedParameter<string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<string>("dbPassword", "");

  if ( dbName_.size() != 0 ) {
    cout << " DB output will go to"
         << " dbName = '" << dbName_ << "'"
         << " dbHostName = '" << dbHostName_ << "'"
         << " dbUserName = '" << dbUserName_ << "'" << endl;
  } else {
    cout << " DB output is disabled" << endl;
  }

  enableSubRun_ = ps.getUntrackedParameter<bool>("enableSubRun", false);

  // location

  location_ =  ps.getUntrackedParameter<string>("location", "H4");

  // base Html output directory

  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", "");

  if ( baseHtmlDir_.size() != 0 ) {
    cout << " HTML output will go to"
         << " baseHtmlDir = '" << baseHtmlDir_ << "'" << endl;
  } else {
    cout << " HTML output is disabled" << endl;
  }

  // collateSources switch

  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  if ( collateSources_ ) {
    cout << " collateSources switch is ON" << endl;
  } else {
    cout << " collateSources switch is OFF" << endl;
  }

  // cloneME switch

  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  if ( cloneME_ ) {
    cout << " cloneME switch is ON" << endl;
  } else {
    cout << " cloneME switch is OFF" << endl;
  }

  // verbosity switch

  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    cout << " verbose switch is ON" << endl;
  } else {
    cout << " verbose switch is OFF" << endl;
  }

  // start DQM user interface instance

  mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_);

  if ( verbose_ ) {
    mui_->setVerbose(1);
  } else {
    mui_->setVerbose(0);
  }

  // will attempt to reconnect upon connection problems (w/ a 5-sec delay)

  mui_->setReconnectDelay(5);

  // global ROOT style

  gStyle->Reset("Default");

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleColor(10);
  gStyle->SetTitleFillColor(10);

  TGaxis::SetMaxDigits(4);

  gStyle->SetOptTitle(kTRUE);
  gStyle->SetTitleX(0.00);
  gStyle->SetTitleY(1.00);
  gStyle->SetTitleW(0.00);
  gStyle->SetTitleH(0.06);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFont(43, "c");
  gStyle->SetTitleFontSize(11);

  gStyle->SetOptStat(kFALSE);
  gStyle->SetStatX(0.99);
  gStyle->SetStatY(0.99);
  gStyle->SetStatW(0.25);
  gStyle->SetStatH(0.20);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatFont(43);
  gStyle->SetStatFontSize(10);

  gStyle->SetOptFit(kFALSE);

  gROOT->ForceStyle();

  // clients' constructors

  integrity_client_      = new EBIntegrityClient(ps, mui_);

  cosmic_client_         = new EBCosmicClient(ps, mui_);
  laser_client_          = new EBLaserClient(ps, mui_);
  pedestal_client_       = new EBPedestalClient(ps, mui_);
  pedestalonline_client_ = new EBPedestalOnlineClient(ps, mui_);
  testpulse_client_      = new EBTestPulseClient(ps, mui_);
  electron_client_       = new EBElectronClient(ps, mui_);

  cout << endl;

}

EcalBarrelMonitorClient::~EcalBarrelMonitorClient(){

  cout << "Exit ..." << endl;

  if ( integrity_client_ ) {
    delete integrity_client_;
  }

  if ( cosmic_client_ ) {
    delete cosmic_client_;
  }
  if ( laser_client_ ) {
    delete laser_client_;
  }
  if ( pedestal_client_ ) {
    delete pedestal_client_;
  }
  if ( pedestalonline_client_ ) {
    delete pedestalonline_client_;
  }
  if ( testpulse_client_ ) {
    delete testpulse_client_;
  }
  if ( electron_client_ ) {
    delete electron_client_;
  }

  this->cleanup();

  sleep(10);

  mui_->disconnect();

  if ( mui_ ) delete mui_;

}

void EcalBarrelMonitorClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->subscribe();

  if ( integrity_client_ ) {
    integrity_client_->beginJob(c);
  }

  if ( cosmic_client_ ) {
    cosmic_client_->beginJob(c);
  }
  if ( laser_client_ ) {
    laser_client_->beginJob(c);
  }
  if ( pedestal_client_ ) {
    pedestal_client_->beginJob(c);
  }
  if ( pedestalonline_client_ ) {
    pedestalonline_client_->beginJob(c);
  }
  if ( testpulse_client_ ) {
    testpulse_client_->beginJob(c);
  }
  if ( electron_client_ ) {
    electron_client_->beginJob(c);
  }

}

void EcalBarrelMonitorClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->beginRunDb();

  if ( integrity_client_ ) {
    integrity_client_->cleanup();
    integrity_client_->beginRun(c);
  }

  if ( cosmic_client_ ) {
    cosmic_client_->cleanup();
    if ( runtype_ == "cosmic" ) {
      cosmic_client_->beginRun(c);
    }
  }
  if ( laser_client_ ) {
    laser_client_->cleanup();
    if ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) {
      laser_client_->beginRun(c);
    }
  }
  if ( pedestal_client_ ) {
    pedestal_client_->cleanup();
    if ( runtype_ == "pedestal" ) {
      pedestal_client_->beginRun(c);
    }
  }
  if ( pedestalonline_client_ ) {
    pedestalonline_client_->cleanup();
    pedestalonline_client_->beginRun(c);
  }
  if ( testpulse_client_ ) {
    testpulse_client_->cleanup();
    if ( runtype_ == "testpulse" ) {
      testpulse_client_->beginRun(c);
    }
  }
  if ( electron_client_ ) {
    electron_client_->cleanup();
    if ( runtype_ == "electron" ) {
      electron_client_->beginRun(c);
    }
  }

}

void EcalBarrelMonitorClient::endJob(void) {

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  if ( integrity_client_ ) {
    integrity_client_->endJob();
  }
  if ( cosmic_client_ ) {
    cosmic_client_->endJob();
  }
  if ( laser_client_ ) {
    laser_client_->endJob();
  }
  if ( pedestal_client_ ) {
    pedestal_client_->endJob();
  }
  if ( pedestalonline_client_ ) {
    pedestalonline_client_->endJob();
  }
  if ( testpulse_client_ ) {
    testpulse_client_->endJob();
  }
  if ( electron_client_ ) {
    electron_client_->endJob();
  }

}

void EcalBarrelMonitorClient::endRun(void) {

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: endRun, jevt = " << jevt_ << endl;

  if ( outputFile_.size() != 0 ) mui_->save(outputFile_);

  this->writeDb();

  if ( baseHtmlDir_.size() != 0 ) this->htmlOutput();

  if ( integrity_client_ ) {
    integrity_client_->endRun();
  }

  if ( cosmic_client_ ) {
    if ( runtype_ == "cosmic" ) {
      cosmic_client_->endRun();
    }
  }
  if ( laser_client_ ) {
    if ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) {
      laser_client_->endRun();
    }
  }
  if ( pedestal_client_ ) {
    if ( runtype_ == "pedestal" ) {
      pedestal_client_->endRun();
    }
  }
  if ( pedestalonline_client_ ) {
    pedestalonline_client_->endRun();
  }
  if ( testpulse_client_ ) {
    if ( runtype_ == "testpulse" ) {
      testpulse_client_->endRun();
    }
  }
  if ( electron_client_ ) {
    if ( runtype_ == "electron" ) {
      electron_client_->endRun();
    }
  }

  this->endRunDb();

  this->cleanup();

  status_  = "unknown";
  run_     = -1;
  evt_     = -1;
  runtype_ = "unknown";

  last_jevt_ = -1;
  last_update_ = 0;

  // this is an effective way to avoid ROOT memory leaks ...

  cout << endl;
  cout << ">>> exit after End-Of-Run <<<" << endl;
  cout << endl;
  this->endJob();
  throw exception();

}

void EcalBarrelMonitorClient::setup(void) {

}

void EcalBarrelMonitorClient::cleanup(void) {

  if ( cloneME_ ) {
    if ( h_ ) delete h_;
  }

  h_ = 0;

}

void EcalBarrelMonitorClient::beginRunDb(void) {

  subrun_ = 0;

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      cout << "Opening DB connection ..." << endl;
      econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_);
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  // create the objects necessary to identify a dataset

  LocationDef locdef;

  locdef.setLocation(location_);

  RunTypeDef rundef;

  rundef.setRunType("TEST");
  if ( runtype_ == "cosmic" ) rundef.setRunType("COSMICS");
  if ( runtype_ == "laser" ) rundef.setRunType("LASER");
  if ( runtype_ == "pedestal" ) rundef.setRunType("PEDESTAL");
  if ( runtype_ == "testpulse" ) rundef.setRunType("TEST_PULSE");
  if ( runtype_ == "electron" ) rundef.setRunType("BEAM");

  rundef.setConfigTag("config_v01");
  rundef.setConfigVersion(1);

  RunTag runtag;

  runtag.setLocationDef(locdef);
  runtag.setRunTypeDef(rundef);
  runtag.setGeneralTag("default");

  Tm startRun;

  startRun.setToCurrentGMTime();
  startRun.setToMicrosTime(startRun.microsTime());

  // setup the RunIOV (on behalf of the DAQ)

  runiov_.setRunNumber(run_);
  runiov_.setRunStart(startRun);
  runiov_.setRunTag(runtag);

  if ( econn ) {
    try {
      cout << "Inserting RunIOV ... " << flush;
      econn->insertRunIOV(&runiov_);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  // fetch the RunIOV back from the DB

  if ( econn ) {
    try {
      cout << "Fetching RunIOV ... " << flush;
      runiov_ = econn->fetchRunIOV(run_);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  location_ = runiov_.getRunTag().getLocationDef().getLocation();

  if ( runiov_.getRunTag().getRunTypeDef().getRunType() == "COSMIC" ) runtype_ = "cosmic";
  if ( runiov_.getRunTag().getRunTypeDef().getRunType() == "LASER" ) runtype_ = "laser";
  if ( runiov_.getRunTag().getRunTypeDef().getRunType() == "PEDESTAL" ) runtype_ = "pedestal";
  if ( runiov_.getRunTag().getRunTypeDef().getRunType() == "TEST_PULSE" ) runtype_ = "testpulse";
  if ( runiov_.getRunTag().getRunTypeDef().getRunType() == "BEAM" ) runtype_ = "electron";

  cout << endl;
  cout << "=============RunIOV:" << endl;
  cout << "Run Number:         " << runiov_.getRunNumber() << endl;
  cout << "Run Start:          " << runiov_.getRunStart().str() << endl;
  cout << "Run End:            " << runiov_.getRunEnd().str() << endl;
  cout << "====================" << endl;
  cout << endl;
  cout << "=============RunTag:" << endl;
  cout << "GeneralTag:         " << runiov_.getRunTag().getGeneralTag() << endl;
  cout << "Location:           " << runiov_.getRunTag().getLocationDef().getLocation() << endl;
  cout << "Run Type:           " << runiov_.getRunTag().getRunTypeDef().getRunType() << endl;
  cout << "Config Tag:         " << runiov_.getRunTag().getRunTypeDef().getConfigTag() << endl;
  cout << "Config Ver:         " << runiov_.getRunTag().getRunTypeDef().getConfigVersion() << endl;
  cout << "====================" << endl;
  cout << endl;

  if ( econn ) {
    try {
      cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EcalBarrelMonitorClient::writeDb(void) {

  subrun_++;

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      cout << "Opening DB connection ..." << endl;
      econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_);
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonVersionDef monverdef;

  monverdef.setMonitoringVersion("test01");

  MonRunTag montag;

  montag.setMonVersionDef(monverdef);
  montag.setGeneralTag("CMSSW");

  Tm startSubRun;

  startSubRun.setToCurrentGMTime();
  startSubRun.setToMicrosTime(startSubRun.microsTime());

  // setup the MonIOV

  moniov_.setRunIOV(runiov_);
  moniov_.setSubRunNumber(subrun_);
  moniov_.setSubRunStart(startSubRun);
  moniov_.setMonRunTag(montag);

  cout << endl;
  cout << "==========MonRunIOV:" << endl;
  cout << "SubRun Number:      " << moniov_.getSubRunNumber() << endl;
  cout << "SubRun Start:       " << moniov_.getSubRunStart().str() << endl;
  cout << "SubRun End:         " << moniov_.getSubRunEnd().str() << endl;
  cout << "====================" << endl;
  cout << endl;
  cout << "==========MonRunTag:" << endl;
  cout << "GeneralTag:         " << moniov_.getMonRunTag().getGeneralTag() << endl;
  cout << "Monitoring Ver:     " << moniov_.getMonRunTag().getMonVersionDef().getMonitoringVersion() << endl;
  cout << "====================" << endl;
  cout << endl;

  int taskl = 0x0;
  int tasko = 0x0;

  if ( integrity_client_ ) {
    if ( status_ == "end-of-run" || ( runtype_ == "cosmic" || runtype_ == "electron" ) ) {
      taskl |= 0x1;
      integrity_client_->writeDb(econn, &moniov_);
      tasko |= 0x0;
    }
  }

  if ( cosmic_client_ ) {
    if ( status_ == "end-of-run" || runtype_ == "cosmic" ) {
      taskl |= 0x1 << 1;
      cosmic_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 1;
    }
  }
  if ( laser_client_ ) {
    if ( status_ == "end-of-run" && ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) ) {
      taskl |= 0x1 << 2;
      laser_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 2;
    }
  }
  if ( pedestal_client_ ) {
    if ( status_ == "end-of-run" && runtype_ == "pedestal" ) {
      taskl |= 0x1 << 3;
      pedestal_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 3;
    }
  }
  if ( pedestalonline_client_ ) {
    if ( status_ == "end-of-run" || ( runtype_ == "cosmic" || runtype_ == "electron" ) ) {
      taskl |= 0x1 << 4;
      pedestalonline_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 4;
    }
  }
  if ( testpulse_client_ ) {
    if ( status_ == "end-of-run" && runtype_ == "testpulse" ) {
      taskl |= 0x1 << 5;
      testpulse_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 5;
    }
  }
  if ( electron_client_ ) {
    if ( status_ == "end-of-run" || runtype_ == "electron" ) {
      taskl |= 0x1 << 6;
      electron_client_->writeDb(econn, &moniov_);
      tasko |= 0x0 << 6;
    }
  }

  EcalLogicID ecid;
  MonRunDat md;
  map<EcalLogicID, MonRunDat> dataset;

  MonRunOutcomeDef monRunOutcomeDef;

  monRunOutcomeDef.setShortDesc("success");

  float nevt = -1.;

  if ( h_ ) nevt = h_->GetEntries();

  md.setNumEvents(int(nevt));
  md.setMonRunOutcomeDef(monRunOutcomeDef);
  md.setRootfileName(outputFile_);
  md.setTaskList(taskl);
  md.setTaskOutcome(tasko);

  cout << "Creating MonRunDatObjects for the database ..." << endl;

  if ( econn ) {
    try {
      ecid = econn->getEcalLogicID("ECAL");
      dataset[ecid] = md;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset, &moniov_);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  cout << endl;

}

void EcalBarrelMonitorClient::endRunDb(void) {

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      cout << "Opening DB connection ..." << endl;
      econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_);
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  EcalLogicID ecid;
  RunDat rd;
  map<EcalLogicID, RunDat> dataset;

  float nevt = -1.;

  if ( h_ ) nevt = h_->GetEntries();

  // setup the RunDat (on behalf of the DAQ)

  rd.setNumEvents(int(nevt));

  cout << "Creating RunDatObjects for the database ..." << endl;

  if ( econn ) {
    try {
      ecid = econn->getEcalLogicID("ECAL");
      dataset[ecid] = rd;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset, &runiov_);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EcalBarrelMonitorClient::subscribe(void){

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: subscribe" << endl;

  // subscribe to monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/STATUS");
  mui_->subscribe("*/EcalBarrel/RUN");
  mui_->subscribe("*/EcalBarrel/EVT");
  mui_->subscribe("*/EcalBarrel/EVTTYPE");
  mui_->subscribe("*/EcalBarrel/RUNTYPE");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EcalBarrelMonitorClient: collate" << endl;

    Char_t histo[80];

    sprintf(histo, "EVTTYPE");
    me_h_ = mui_->collate1D(histo, histo, "EcalBarrel/Sums");
    sprintf(histo, "*/EcalBarrel/EVTTYPE");
    mui_->add(me_h_, histo);

  }

}

void EcalBarrelMonitorClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/STATUS");
  mui_->subscribeNew("*/EcalBarrel/RUN");
  mui_->subscribeNew("*/EcalBarrel/EVT");
  mui_->subscribeNew("*/EcalBarrel/EVTTYPE");
  mui_->subscribeNew("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::unsubscribe(void) {

  if ( verbose_ ) cout << "EcalBarrelMonitorClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EcalBarrelMonitorClient: uncollate" << endl;

    if ( mui_ ) {

      mui_->removeCollate(me_h_);

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/STATUS");
  mui_->unsubscribe("*/EcalBarrel/RUN");
  mui_->unsubscribe("*/EcalBarrel/EVT");
  mui_->unsubscribe("*/EcalBarrel/EVTTYPE");
  mui_->unsubscribe("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EcalBarrelMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  bool stay_in_loop = mui_->update();

  this->subscribeNew();

  if ( integrity_client_ ) {
    integrity_client_->subscribeNew();
  }

  if ( cosmic_client_ ) {
    if ( runtype_ == "cosmic" ) {
      cosmic_client_->subscribeNew();
    }
  }
  if ( laser_client_ ) {
    if ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) {
      laser_client_->subscribeNew();
    }
  }
  if ( pedestal_client_ ) {
    if ( runtype_ == "pedestal" ) {
      pedestal_client_->subscribeNew();
    }
  }
  if ( pedestalonline_client_ ) {
    pedestalonline_client_->subscribeNew();
  }
  if ( testpulse_client_ ) {
    if ( runtype_ == "testpulse" ) {
      testpulse_client_->subscribeNew();
    }
  }
  if ( electron_client_ ) {
    if ( runtype_ == "electron" ) {
      electron_client_->subscribeNew();
    }
  }

  // # of full monitoring cycles processed
  int updates = mui_->getNumUpdates();

  Char_t histo[150];

  MonitorElement* me;
  string s;

  bool update = false;

  if ( stay_in_loop && updates != last_update_ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/STATUS");
    me = mui_->get(histo);
    if ( me ) {
      s = me->valueString();
      status_ = "unknown";
      if ( s.substr(2,1) == "0" ) status_ = "begin-of-run";
      if ( s.substr(2,1) == "1" ) status_ = "running";
      if ( s.substr(2,1) == "2" ) status_ = "end-of-run";
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/RUN");
    me = mui_->get(histo);
    if ( me ) {
      s = me->valueString();
      run_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EVT");
    me = mui_->get(histo);
    if ( me ) {
      s = me->valueString();
      evt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &evt_);
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EVTTYPE");
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EVTTYPE");
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( h_ ) delete h_;
          sprintf(histo, "ME EVTTYPE");
          h_ = dynamic_cast<TH1F*> ((ob->operator->())->Clone(histo));
        } else {
          h_ = dynamic_cast<TH1F*> (ob->operator->());
        }
      }
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUNTYPE");
    if ( me ) {
      s = me->valueString();
      runtype_ = "unknown";
      if ( s.substr(2,1) == "0" ) runtype_ = "cosmic";
      if ( s.substr(2,1) == "1" ) runtype_ = "laser";
      if ( s.substr(2,1) == "2" ) runtype_ = "pedestal";
      if ( s.substr(2,1) == "3" ) runtype_ = "testpulse";
      if ( s.substr(2,1) == "4" ) runtype_ = "electron";
    }

    if ( verbose_ ) cout << " updates = "  << updates << endl;

    cout << " run = "      << run_      <<
            " event = "    << evt_      <<
            " status = "   << status_   << endl;

    cout << " runtype = "  << runtype_  <<
            " location = " << location_ << endl;

    if ( h_ ) {
      cout << " event type = " << flush;
      for ( int i = 1; i <= 10; i++ ) {
        cout << h_->GetBinContent(i) << " " << flush;
      }
      if ( h_->GetEntries() != 0 ) {
        cout << "  ( " << flush;
        if ( h_->GetBinContent(1) != 0 ) cout << "cosmic " << flush;
        if ( h_->GetBinContent(2) != 0 ) cout << "laser " << flush;
        if ( h_->GetBinContent(3) != 0 ) cout << "pedestal " << flush;
        if ( h_->GetBinContent(4) != 0 ) cout << "testpulse " << flush;
        if ( h_->GetBinContent(5) != 0 ) cout << "electron " << flush;
        cout << ")" << flush;
      }
      cout << endl;
    }

    update = true;

    last_update_ = updates;

    last_jevt_ = jevt_;

  }

  if ( status_ == "unknown" ) {

    if ( update ) unknowns_++;

    if ( unknowns_ >= 10 ) {

      cout << "Too many 'unknown' states ..." << endl;

      cout << "Forcing begin-of-job ... NOW !" << endl;

      this->beginJob(c);

    }

  }

  if ( status_ == "begin-of-run" ) {

    if ( ! begin_run_done_ ) {

      this->beginRun(c);
      begin_run_done_ = true;
      forced_begin_run_ = false;

      end_run_done_ = false;

    }

  }

  if ( status_ == "running" ) {

    if ( ! begin_run_done_ && ! end_run_done_ && ! forced_end_run_ ) {

      if ( run_ > 0 && evt_ > 0 && runtype_ != "unknown" ) {

        cout << "Running with no begin_run ..." << endl;

        cout << "Forcing begin-of-run ... NOW !" << endl;

        status_ = "begin-of-run";
        this->beginRun(c);
        begin_run_done_ = true;
        forced_begin_run_ = true;

        end_run_done_ = false;

      }

    }

    if ( begin_run_done_ && ! forced_begin_run_ && ! end_run_done_ ) {

      if ( run_ > 0 && evt_ > 0 && runtype_ != "unknown" ) {

        if ( ( jevt_ - last_jevt_ ) > 200 ) {

          cout << "Running with no updates since too long ..." << endl;

          cout << "Forcing end-of-run ... NOW !" << endl;

          begin_run_done_ = false;

          status_ = "end-of-run";
          this->endRun();
          end_run_done_ = true;
          forced_end_run_ = true;

        }

      }

    }

    if ( begin_run_done_ && ! end_run_done_ ) {

      if ( update && updates % 5 == 0 ) {

        if ( integrity_client_ ) {
          integrity_client_->analyze(e, c);
        }

        if ( cosmic_client_ ) {
          if ( h_ && h_->GetBinContent(1) != 0 ) {
            if ( runtype_ == "cosmic" ) {
              cosmic_client_->analyze(e, c);
            }
          }
        }
        if ( laser_client_ ) {
          if ( h_ && h_->GetBinContent(2) != 0 ) {
            if ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) {
              laser_client_->analyze(e, c);
            }
          }
        }
        if ( pedestal_client_ ) {
          if ( h_ && h_->GetBinContent(3) != 0 ) {
            if ( runtype_ == "pedestal" ) {
              pedestal_client_->analyze(e, c);
            }
          }
        }
        if ( pedestalonline_client_ ) {
          pedestalonline_client_->analyze(e, c);
        }
        if ( testpulse_client_ ) {
          if ( h_ && h_->GetBinContent(4) != 0 ) {
            if ( runtype_ == "testpulse" ) {
              testpulse_client_->analyze(e, c);
            }
          }
        }
        if ( electron_client_ ) {
          if ( h_ && h_->GetBinContent(5) != 0 ) {
            if ( runtype_ == "electron" ) {
              electron_client_->analyze(e, c);
            }
          }
        }

        if ( enableSubRun_ ) {
          if ( update && updates % 10 == 0 ) {
            if ( runtype_ == "cosmic" || runtype_ == "electron" ) this->writeDb();
          }
        }

      }

    }

  }

  if ( status_ == "end-of-run" ) {

    if ( begin_run_done_ && ! end_run_done_ ) {

      if ( integrity_client_ ) {
        integrity_client_->analyze(e, c);
      }

      if ( cosmic_client_ ) {
        if ( h_ && h_->GetBinContent(1) != 0 ) {
          if ( runtype_ == "cosmic" ) {
            cosmic_client_->analyze(e, c);
          }
        }
      }
      if ( laser_client_ ) {
        if ( h_ && h_->GetBinContent(2) != 0 ) {
          if ( runtype_ == "cosmic" || runtype_ == "laser" || runtype_ == "electron" ) {
            laser_client_->analyze(e, c);
          }
        }
      }
      if ( pedestal_client_ ) {
        if ( h_ && h_->GetBinContent(3) != 0 ) {
          if ( runtype_ == "pedestal" ) {
            pedestal_client_->analyze(e, c);
          }
        }
      }
      if ( pedestalonline_client_ ) {
        pedestalonline_client_->analyze(e, c);
      }
      if ( testpulse_client_ ) {
        if ( h_ && h_->GetBinContent(4) != 0 ) {
          if ( runtype_ == "testpulse" ) {
            testpulse_client_->analyze(e, c);
          }
        }
      }
      if ( electron_client_ ) {
        if ( h_ && h_->GetBinContent(5) != 0 ) {
          if ( runtype_ == "electron" ) {
            electron_client_->analyze(e, c);
          }
        }
      }

      begin_run_done_ = false;

      this->endRun();
      end_run_done_ = true;
      forced_end_run_ = false;

    }

  }

}

void EcalBarrelMonitorClient::htmlOutput(void){

  cout << "Preparing EcalBarrelMonitorClient html output ..." << endl;

  char tmp[10];

  sprintf(tmp, "%09d", run_);

  string htmlDir = baseHtmlDir_ + "/" + tmp + "/";

  system(("/bin/mkdir -p " + htmlDir).c_str());

  ofstream htmlFile;

  htmlFile.open((htmlDir + "index.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Executed Tasks index</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Executed tasks for run:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << run_ <<"</span></h2> " << endl;
  htmlFile << "<h2>Run type:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << runtype_ <<"</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<ul>" << endl;

  string htmlName;

  // Integrity check

  if ( h_ && h_->GetEntries() != 0 ) {
    if ( integrity_client_ ) {
      htmlName = "EBIntegrityClient.html";
      integrity_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Data Integrity</a></li>" << endl;
    }
  }

  // Cosmic check

  if ( h_ && h_->GetBinContent(1) != 0 ) {
    if ( cosmic_client_ ) {
      htmlName = "EBCosmicClient.html";
      cosmic_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Cosmic</a></li>" << endl;
    }
  }

  // Laser check

  if ( h_ && h_->GetBinContent(2) != 0 ) {
    if ( laser_client_ ) {
      htmlName = "EBLaserClient.html";
      laser_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Laser</a></li>" << endl;
    }
  }

  // Pedestal check (normal)

  if ( h_ && h_->GetBinContent(3) != 0 ) {
    if ( pedestal_client_ ) {
      htmlName = "EBPedestalClient.html";
      pedestal_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Pedestal</a></li>" << endl;
    }
  }

  // Pedestal check (pre-sample)

  if ( h_ && h_->GetEntries() != 0 ) {
    if ( pedestalonline_client_ ) {
      htmlName = "EBPedestalOnlineClient.html";
      pedestalonline_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Pedestal Online</a></li>" << endl;
    }
  }

  // Test pulse check

  if ( h_ && h_->GetBinContent(4) != 0 ) {
    if ( testpulse_client_ ) {
      htmlName = "EBTestPulseClient.html";
      testpulse_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Test pulse</a></li>" << endl;
    }
  }

  // Electron check

  if ( h_ && h_->GetBinContent(5) != 0 ) {
    if ( electron_client_ ) {
      htmlName = "EBElectronClient.html";
      electron_client_->htmlOutput(run_, htmlDir, htmlName);
      htmlFile << "<li><a href=\"" << htmlName << "\">Electron</a></li>" << endl;
    }
  }

  htmlFile << "</ul>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  cout << endl;

}

