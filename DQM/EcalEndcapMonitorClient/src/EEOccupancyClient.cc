/*
 * \file EEOccupancyClient.cc
 *
 * $Date: 2008/01/26 20:13:56 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEOccupancyClient::EEOccupancyClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableMonitorDaemon_ switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjEta_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjEta_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjEta_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjEta_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

EEOccupancyClient::~EEOccupancyClient(){

}

void EEOccupancyClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EEOccupancyClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEOccupancyClient::beginRun(void){

  if ( verbose_ ) cout << "EEOccupancyClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEOccupancyClient::endJob(void) {

  if ( verbose_ ) cout << "EEOccupancyClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::endRun(void) {

  if ( verbose_ ) cout << "EEOccupancyClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::setup(void) {

  char histo[200];

  dbe_->setCurrentFolder( "EcalEndcap/EEOccupancyClient" );

}

void EEOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( int i=0; i<3; ++i ) {
      if ( h01_[0][i] ) delete h01_[0][i];
      if ( h01ProjEta_[0][i] ) delete h01ProjEta_[0][i];
      if ( h01ProjPhi_[0][i] ) delete h01ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjEta_[1][i] ) delete h01ProjEta_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[0][i] ) delete h02_[0][i];
      if ( h02ProjEta_[0][i] ) delete h02ProjEta_[0][i];
      if ( h02ProjPhi_[0][i] ) delete h02ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjEta_[1][i] ) delete h01ProjEta_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }
      
  }

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjEta_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjEta_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjEta_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjEta_[1][i] = 0;
    h02ProjPhi_[1][i] = 0;
  }

}

bool EEOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EEOccupancyClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

}

void EEOccupancyClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EEOccupancyClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:OccupancyTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">OCCUPANCY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "WORK IN PROGRESS" << std::endl;
  return;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

