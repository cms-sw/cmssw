/*
 * \file EcalBarrelMonitorClient.cc
 * 
 * $Date: 2005/11/26 15:38:21 $
 * $Revision: 1.44 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h>

EcalBarrelMonitorClient::EcalBarrelMonitorClient(const edm::ParameterSet& ps){

  cout << endl;
  cout << " *** Ecal Barrel Generic Monitor Client ***" << endl;
  cout << endl;

  init_run_done_ = false;

  mui_ = 0;
  econn_ = 0;

  h_ = 0;

  // default client name
  clientName_ = ps.getUntrackedParameter<string>("clientName", "EcalBarrelMonitorClient");

  // default collector host name
  hostName_ = ps.getUntrackedParameter<string>("hostName", "localhost");

  // default host port
  hostPort_ = ps.getUntrackedParameter<int>("hostPort", 9090);;

  cout << " Client " << clientName_
       << " begins requesting monitoring from host " << hostName_
       << " on port " << hostPort_ << endl;

  // start user interface instance
  mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_);

  mui_->setVerbose(1);

  // will attempt to reconnect upon connection problems (w/ a 5-sec delay)
  mui_->setReconnectDelay(5);

  integrity_client_ = new EBIntegrityClient(ps, mui_);
  laser_client_ = new EBLaserClient(ps, mui_);
  pndiode_client_ = new EBPnDiodeClient(ps, mui_);
  pedestal_client_ = new EBPedestalClient(ps, mui_);
  pedpresample_client_ = new EBPedPreSampleClient(ps, mui_);
  testpulse_client_ = new EBTestPulseClient(ps, mui_);

  cosmic_client_ = new EBCosmicClient(ps, mui_);

  // Ecal Cond DB
  dbName_ = ps.getUntrackedParameter<string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<string>("dbHostName", "");
  dbUserName_ = ps.getUntrackedParameter<string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<string>("dbPassword", "");

  if ( dbName_.size() != 0 ) {
    cout << " DB output will go to"
         << " dbName = " << dbName_
         << " dbHostName = " << dbHostName_
         << " dbUserName = " << dbUserName_ << endl;
  } else {
    cout << " DB output is disabled" << endl;
  }

  // base Html output directory
  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", ".");

  if ( baseHtmlDir_.size() != 0 ) {
    cout << " HTML output will go to"
         << " baseHtmlDir = " << baseHtmlDir_ << endl;
  } else {
    cout << " HTML output is disabled" << endl;
  }

}

EcalBarrelMonitorClient::~EcalBarrelMonitorClient(){

  cout << "Exit ..." << endl;

  if ( h_ ) delete h_;

  if ( integrity_client_ ) delete integrity_client_;
  if ( laser_client_ ) delete laser_client_;
  if ( pndiode_client_ ) delete pndiode_client_;
  if ( pedestal_client_ ) delete pedestal_client_;
  if ( pedpresample_client_ ) delete pedpresample_client_;
  if ( testpulse_client_ ) delete testpulse_client_;

  if ( cosmic_client_ ) delete cosmic_client_;

  usleep(100);

  if ( mui_ ) delete mui_;

}

void EcalBarrelMonitorClient::beginJob(const edm::EventSetup& c){

  cout << "EcalBarrelMonitorClient: beginJob" << endl;

  ievt_ = 0;

  this->subscribe();

  integrity_client_->beginJob(c);
  laser_client_->beginJob(c);
  pndiode_client_->beginJob(c);
  pedestal_client_->beginJob(c);
  pedpresample_client_->beginJob(c);
  testpulse_client_->beginJob(c);

  cosmic_client_->beginJob(c);

  this->beginRun(c);
  init_run_done_ = true;

}

void EcalBarrelMonitorClient::beginRun(const edm::EventSetup& c){
  
  cout << "EcalBarrelMonitorClient: beginRun" << endl;

  jevt_ = 0;

  last_jevt_ = -1;

  last_update_ = 0;

  if ( h_ ) delete h_;
  h_ = 0;

  status_  = "unknown";
  run_     = 0;
  evt_     = 0;
  runtype_ = "unknown";

  integrity_client_->beginRun(c);
  laser_client_->beginRun(c);
  pndiode_client_->beginRun(c);
  pedestal_client_->beginRun(c);
  pedpresample_client_->beginRun(c);
  testpulse_client_->beginRun(c);

  cosmic_client_->beginRun(c);

}

void EcalBarrelMonitorClient::endJob(void) {

  cout << "EcalBarrelMonitorClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  integrity_client_->endJob();
  laser_client_->endJob();
  pndiode_client_->endJob();
  pedestal_client_->endJob();
  pedpresample_client_->endJob();
  testpulse_client_->endJob();

  cosmic_client_->endJob();

}

void EcalBarrelMonitorClient::endRun(void) {

  cout << "EcalBarrelMonitorClient: endRun, jevt = " << jevt_ << endl;

  mui_->save("EcalBarrelMonitorClient.root");

  econn_ = 0;

  if ( dbName_.size() != 0 ) {
    try {
      cout << "Opening DB connection." << endl;
      econn_ = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_);
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  // The objects necessary to identify a dataset
  runiov_ = new RunIOV();
  runtag_ = new RunTag();

  Tm startTm;

  // Set the beginning time
  startTm.setToCurrentGMTime();
  startTm.setToMicrosTime(startTm.microsTime());

  cout << "Setting run " << run_ << " start_time " << startTm.str() << endl;

  runiov_->setRunNumber(run_);
  runiov_->setRunStart(startTm);

  runtag_->setRunType(runtype_);
  runtag_->setLocation(location_);
  runtag_->setMonitoringVersion("version 1");

  EcalLogicID ecid;
  RunDat r;
  map<EcalLogicID, RunDat> dataset;

  cout << "Writing RunDatObjects to database ..." << endl;

  float nevt = 0.;

  if ( h_ ) nevt = h_->GetEntries();

  r.setNumEvents(int(nevt));

  if ( econn_ ) {
    try {
      int ism = 1;
      ecid = econn_->getEcalLogicID("EB_supermodule", ism);
      dataset[ecid] = r;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn_ ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn_->insertDataSet(&dataset, runiov_, runtag_ );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  integrity_client_->endRun(econn_, runiov_, runtag_);
  laser_client_->endRun(econn_, runiov_, runtag_);
  pndiode_client_->endRun(econn_, runiov_, runtag_);
  pedestal_client_->endRun(econn_, runiov_, runtag_);
  pedpresample_client_->endRun(econn_, runiov_, runtag_);
  testpulse_client_->endRun(econn_, runiov_, runtag_);

  cosmic_client_->endRun(econn_, runiov_, runtag_);

  if ( econn_ ) {
    try {
      cout << "Closing DB connection." << endl;
      delete econn_;
      econn_ = 0;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( runiov_ ) delete runiov_;
  if ( runtag_ ) delete runtag_;

  if ( baseHtmlDir_.size() != 0 ) this->htmlOutput();

}

void EcalBarrelMonitorClient::subscribe(void){

  // subscribe to monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/STATUS");
  mui_->subscribe("*/EcalBarrel/RUN");
  mui_->subscribe("*/EcalBarrel/EVT");
  mui_->subscribe("*/EcalBarrel/EVTTYPE");
  mui_->subscribe("*/EcalBarrel/RUNTYPE");

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

  // subscribe to all monitorable matching pattern 
  mui_->unsubscribe("*/EcalBarrel/STATUS");
  mui_->unsubscribe("*/EcalBarrel/RUN");
  mui_->unsubscribe("*/EcalBarrel/EVT");
  mui_->unsubscribe("*/EcalBarrel/EVTTYPE");
  mui_->unsubscribe("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )
    cout << "EcalBarrelMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  bool stay_in_loop = mui_->update();

  this->subscribeNew();

  integrity_client_->subscribeNew();
  laser_client_->subscribeNew();
  pndiode_client_->subscribeNew();
  pedestal_client_->subscribeNew();
  pedpresample_client_->subscribeNew();
  testpulse_client_->subscribeNew();

  cosmic_client_->subscribeNew();

  // # of full monitoring cycles processed
  int updates = mui_->getNumUpdates();

  if ( stay_in_loop && updates != last_update_ ) {

    MonitorElement* me;
    string s;

    me = mui_->get("Collector/FU0/EcalBarrel/STATUS");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) status_ = "begin-of-run";
      if ( s.substr(2,1) == "1" ) status_ = "running";
      if ( s.substr(2,1) == "2" ) status_ = "end-of-run";
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUN");
    if ( me ) {
      s = me->valueString();
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
    }

    me = mui_->get("Collector/FU0/EcalBarrel/EVT");
    if ( me ) {
      s = me->valueString();
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &evt_);
    }

    me = mui_->get("Collector/FU0/EcalBarrel/EVTTYPE");
    if ( me ) {
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h_ ) delete h_;
        h_ = dynamic_cast<TH1F*> ((ob->operator->())->Clone("ME EVTTYPE"));
      }
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUNTYPE");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) runtype_ = "cosmic";
      if ( s.substr(2,1) == "1" ) runtype_ = "laser";
      if ( s.substr(2,1) == "2" ) runtype_ = "pedestal";
      if ( s.substr(2,1) == "3" ) runtype_ = "testpulse";
    }

    location_ = "H4";

    cout << " run = "      << run_      <<
            " event = "    << evt_      << endl;
    cout << " updates = "  << updates   <<
            " status = "   << status_   <<
            " runtype = "  << runtype_  <<
            " location = " << location_ << endl;

    if ( h_ ) {
      cout << " event type = " << flush;
      for ( int i = 1; i <=10; i++ ) {
        cout << h_->GetBinContent(i) << " " << flush;
      }
      cout << endl;
    }

    if ( status_ == "unknown" ) {
      init_run_done_ = false;
    }

    if ( status_ == "begin-of-run" ) {
      if ( ! init_run_done_ ) {
        this->beginRun(c);
        init_run_done_ = true;
      }
    }

    if ( status_ == "running" ) {
      init_run_done_ = false;
      if ( last_update_ == 0 || updates % 5 == 0 ) {
                                               integrity_client_->analyze(e, c);
        if ( h_ && h_->GetBinContent(2) != 0 ) laser_client_->analyze(e, c);
        if ( h_ && h_->GetBinContent(2) != 0 ) pndiode_client_->analyze(e, c);
        if ( h_ && h_->GetBinContent(3) != 0 ) pedestal_client_->analyze(e, c);
                                               pedpresample_client_->analyze(e, c);
        if ( h_ && h_->GetBinContent(4) != 0 ) testpulse_client_->analyze(e, c);

        if ( h_ && h_->GetBinContent(1) != 0 ) cosmic_client_->analyze(e, c);
      }
    }

    if ( status_ == "end-of-run" ) {
      init_run_done_ = false;
                                             integrity_client_->analyze(e, c);
      if ( h_ && h_->GetBinContent(2) != 0 ) laser_client_->analyze(e, c);
      if ( h_ && h_->GetBinContent(2) != 0 ) pndiode_client_->analyze(e, c);
      if ( h_ && h_->GetBinContent(3) != 0 ) pedestal_client_->analyze(e, c);
                                             pedpresample_client_->analyze(e, c);
      if ( h_ && h_->GetBinContent(4) != 0 ) testpulse_client_->analyze(e, c);

      if ( h_ && h_->GetBinContent(1) != 0 ) cosmic_client_->analyze(e, c);
      this->endRun();
    }

    if ( updates % 100 == 0 ) {
      mui_->save("EcalBarrelMonitorClient.root");
    }

    last_update_ = updates;

    last_jevt_ = jevt_;

  }

  if ( run_ != 0 &&
       evt_ != 0 &&
       status_ == "running" &&
       jevt_ - last_jevt_ > 200 ) {

    cout << "Running with no updates since too long ..." << endl;

    cout << "Forcing end-of-run ... NOW !" << endl;

                                           integrity_client_->analyze(e, c);
    if ( h_ && h_->GetBinContent(2) != 0 ) laser_client_->analyze(e, c);
    if ( h_ && h_->GetBinContent(2) != 0 ) pndiode_client_->analyze(e, c);
    if ( h_ && h_->GetBinContent(3) != 0 ) pedestal_client_->analyze(e, c); 
                                           pedpresample_client_->analyze(e, c);
    if ( h_ && h_->GetBinContent(4) != 0 ) testpulse_client_->analyze(e, c);

    if ( h_ && h_->GetBinContent(1) != 0 ) cosmic_client_->analyze(e, c);

    this->endRun();

    cout << "Forcing start-of-run ... NOW !" << endl;

    this->beginRun(c);
    init_run_done_ = true;

  }

//  usleep(1000);

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

  htmlName = "EBIntegrityClient.html";
  integrity_client_->htmlOutput(run_, htmlDir, htmlName);
  htmlFile << "<li><a href=\"" << htmlName << "\">Data Integrity</a></li>" << endl;

  // Laser check

  if ( h_ && h_->GetBinContent(2) != 0 ) {
    htmlName = "EBLaserClient.html";
    laser_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<li><a href=\"" << htmlName << "\">Laser</a></li>" << endl;
  }

  // PNs check

  if ( h_ && h_->GetBinContent(2) != 0 ) {
    htmlName = "EBPnDiodeClient.html";
    pndiode_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<li><a href=\"" << htmlName << "\">PNdiode</a></li>" << endl;
  }

  // Pedestal check (normal)

  if ( h_ && h_->GetBinContent(3) != 0 ) {
    htmlName = "EBPedestalClient.html";
    pedestal_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<li><a href=\"" << htmlName << "\">Pedestal</a></li>" << endl;
  }

  // Pedestal check (pre-sample)

  htmlName = "EBPedPreSampleClient.html";
  pedpresample_client_->htmlOutput(run_, htmlDir, htmlName);
  htmlFile << "<li><a href=\"" << htmlName << "\">Pedestal on Presample</a></li>" << endl;

  // Test pulse check

  if ( h_ && h_->GetBinContent(4) != 0 ) {
    htmlName = "EBTestPulseClient.html";
    testpulse_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<li><a href=\"" << htmlName << "\">Test pulse</a></li>" << endl;
  }

  // Cosmic check

  if ( h_ && h_->GetBinContent(1) != 0 ) {
    htmlName = "EBCosmicClient.html";
    cosmic_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<li><a href=\"" << htmlName << "\">Cosmic</a></li>" << endl;
  }

  htmlFile << "</ul>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

