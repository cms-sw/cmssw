/*
 * \file EBBeamClient.cc
 *
 * $Date: 2006/06/17 09:44:36 $
 * $Revision: 1.8 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBBeamClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

EBBeamClient::EBBeamClient(const ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i < 37; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

}

EBBeamClient::~EBBeamClient(){

}

void EBBeamClient::beginJob(void){

  if ( verbose_ ) cout << "EBBeamClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBBeamClient::beginRun(void){

  if ( verbose_ ) cout << "EBBeamClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBBeamClient::endJob(void) {

  if ( verbose_ ) cout << "EBBeamClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBBeamClient::endRun(void) {

  if ( verbose_ ) cout << "EBBeamClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBBeamClient::setup(void) {

}

void EBBeamClient::cleanup(void) {

}

void EBBeamClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov, int ism) {

  EcalLogicID ecid;
  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  if ( econn ) {
    try {
      cout << "Inserting MonOccupancyDat ..." << flush;
      if ( dataset.size() != 0 ) econn->insertDataSet(&dataset, moniov );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBBeamClient::subscribe(void){

  if ( verbose_ ) cout << "EBBeamClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamClient: collate" << endl;

  }

}

void EBBeamClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern

}

void EBBeamClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBBeamClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBBeamClient: uncollate" << endl;

  }

  // unsubscribe to all monitorable matching pattern

}

void EBBeamClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBBeamClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

}

void EBBeamClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBBeamClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:BeamTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Beam</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

