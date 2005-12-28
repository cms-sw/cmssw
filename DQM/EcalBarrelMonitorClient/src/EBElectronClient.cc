/*
 * \file EBElectronClient.cc
 * 
 * $Date: 2005/12/26 13:14:26 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBElectronClient.h>

EBElectronClient::EBElectronClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBElectronClient::~EBElectronClient(){

}

void EBElectronClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBElectronClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBElectronClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBElectronClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBElectronClient::endJob(void) {

  if ( verbose_ ) cout << "EBElectronClient: endJob, ievt = " << ievt_ << endl;

}

void EBElectronClient::endRun(void) {

  if ( verbose_ ) cout << "EBElectronClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBElectronClient::setup(void) {

}

void EBElectronClient::cleanup(void) {

}

void EBElectronClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
//      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBElectronClient::subscribe(void){

  if ( verbose_ ) cout << "EBElectronClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBElectronClient: collate" << endl;

  }

}

void EBElectronClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern

}

void EBElectronClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBElectronClient: unsubscribe" << endl;

  if ( collateSources_ ) {
  
    if ( verbose_ ) cout << "EBElectronClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) { 

    }

  }

  // unsubscribe to all monitorable matching pattern

}

void EBElectronClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBElectronClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

}

void EBElectronClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBElectronClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:ElectronTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Electron</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .jpg files from existing histograms

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

