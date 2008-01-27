/*
 * \file EEOccupancyClient.cc
 *
 * $Date: 2008/01/26 22:47:24 $
 * $Revision: 1.7 $
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
    h01ProjR_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjR_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjR_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjR_[1][i] = 0;
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

  dbe_->setCurrentFolder( "EcalEndcap/EEOccupancyClient" );

}

void EEOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( int i=0; i<3; ++i ) {
      if ( h01_[0][i] ) delete h01_[0][i];
      if ( h01ProjR_[0][i] ) delete h01ProjR_[0][i];
      if ( h01ProjPhi_[0][i] ) delete h01ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjR_[1][i] ) delete h01ProjR_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[0][i] ) delete h02_[0][i];
      if ( h02ProjR_[0][i] ) delete h02ProjR_[0][i];
      if ( h02ProjPhi_[0][i] ) delete h02ProjPhi_[0][i];
      if ( h01_[1][i] ) delete h01_[1][i];
      if ( h01ProjR_[1][i] ) delete h01ProjR_[1][i];
      if ( h01ProjPhi_[1][i] ) delete h01ProjPhi_[1][i];
    }
      
  }

  for ( int i=0; i<3; i++) {
    h01_[0][i] = 0;
    h01ProjR_[0][i] = 0;
    h01ProjPhi_[0][i] = 0;
    h01_[1][i] = 0;
    h01ProjR_[1][i] = 0;
    h01ProjPhi_[1][i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[0][i] = 0;
    h02ProjR_[0][i] = 0;
    h02ProjPhi_[0][i] = 0;
    h02_[1][i] = 0;
    h02ProjR_[1][i] = 0;
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

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE -").c_str());
  me = dbe_->get(histo);
  h01_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE - projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE - projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE +").c_str());
  me = dbe_->get(histo);
  h01_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE + projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE + projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE -").c_str());
  me = dbe_->get(histo);
  h01_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE +").c_str());
  me = dbe_->get(histo);
  h01_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE -").c_str());
  me = dbe_->get(histo);
  h01_[0][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE - projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE - projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE +").c_str());
  me = dbe_->get(histo);
  h01_[1][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE + projection R").c_str());
  me = dbe_->get(histo);
  h01ProjR_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE + projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE -").c_str());
  me = dbe_->get(histo);
  h02_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection R").c_str());
  me = dbe_->get(histo);
  h02ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE +").c_str());
  me = dbe_->get(histo);
  h02_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection R").c_str());
  me = dbe_->get(histo);
  h02ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE -").c_str());
  me = dbe_->get(histo);
  h02_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE - projection R").c_str());
  me = dbe_->get(histo);
  h02ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE - projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE +").c_str());
  me = dbe_->get(histo);
  h02_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE + projection R").c_str());
  me = dbe_->get(histo);
  h02ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEOccupancyTask/EEOT TP thr digi occupancy EE + projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][1] );

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

  const int csize1D = 250;
  const int csize2D = 500;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C labelGrid1("labelGrid1","label grid for EE -", 10, -150.0, 150.0, 10, -150.0, 150.0);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid1.SetBinContent(i, j, -10);
    }
  }

  labelGrid1.SetBinContent(2, 5, -3);
  labelGrid1.SetBinContent(2, 7, -2);
  labelGrid1.SetBinContent(4, 9, -1);
  labelGrid1.SetBinContent(7, 9, -9);
  labelGrid1.SetBinContent(9, 7, -8);
  labelGrid1.SetBinContent(9, 5, -7);
  labelGrid1.SetBinContent(8, 3, -6);
  labelGrid1.SetBinContent(5, 2, -5);
  labelGrid1.SetBinContent(3, 3, -4);

  labelGrid1.SetMarkerSize(2);
  labelGrid1.SetMinimum(-9.01);
  labelGrid1.SetMaximum(-0.01);

  TH2C labelGrid2("labelGrid2","label grid for EE +", 10, -150.0, 150.0, 10, -150.0, 150.0);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      labelGrid2.SetBinContent(i, j, -10);
    }
  }

  labelGrid2.SetBinContent(2, 5, +7);
  labelGrid2.SetBinContent(2, 7, +8);
  labelGrid2.SetBinContent(4, 9, +9);
  labelGrid2.SetBinContent(7, 9, +1);
  labelGrid2.SetBinContent(9, 7, +2);
  labelGrid2.SetBinContent(9, 5, +3);
  labelGrid2.SetBinContent(8, 3, +4);
  labelGrid2.SetBinContent(6, 2, +5);
  labelGrid2.SetBinContent(3, 3, +6);

  labelGrid2.SetMarkerSize(2);
  labelGrid2.SetMinimum(+0.01);
  labelGrid2.SetMaximum(+9.01);

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

