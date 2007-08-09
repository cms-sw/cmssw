/*
 * \file EETriggerTowerClient.cc
 *
 * $Date: 2007/07/29 07:18:20 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <bitset>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerClient::EETriggerTowerClient(const ParameterSet& ps){

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // enableQT switch
  enableQT_ = ps.getUntrackedParameter<bool>("enableQT", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i < 19; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    i01_[ism-1] = 0;
    j01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mej01_[ism-1] = 0;

    for (int j = 0; j < 68 ; j++) {

      k01_[ism-1][j] = 0;
      k02_[ism-1][j] = 0;

      mek01_[ism-1][j] = 0;
      mek02_[ism-1][j] = 0;

    }

  }

}

EETriggerTowerClient::~EETriggerTowerClient(){

}

void EETriggerTowerClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EETriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EETriggerTowerClient::beginRun(void){

  if ( verbose_ ) cout << "EETriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EETriggerTowerClient::endJob(void) {

  if ( verbose_ ) cout << "EETriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETriggerTowerClient::endRun(void) {

  if ( verbose_ ) cout << "EETriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETriggerTowerClient::setup(void) {

}

void EETriggerTowerClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( j01_[ism-1] ) delete j01_[ism-1];
    }

    h01_[ism-1] = 0;
    i01_[ism-1] = 0;
    j01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mej01_[ism-1] = 0;

    for ( int j = 0; j < 68 ; j++ ) {

      if ( cloneME_ ) {
        if ( k01_[ism-1][j] ) delete k01_[ism-1][j];
        if ( k02_[ism-1][j] ) delete k02_[ism-1][j];
      }

      k01_[ism-1][j] = 0;
      k02_[ism-1][j] = 0;

      mek01_[ism-1][j] = 0;
      mek02_[ism-1][j] = 0;

    }

  }

}

bool EETriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EETriggerTowerClient::subscribe(void){

  if ( verbose_ ) cout << "EETriggerTowerClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Et map SM%02d", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT FineGrainVeto SM%02d", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Flags SM%02d", ism);
    mui_->subscribe(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
    }

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETriggerTowerClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EETTT Et map SM%02d", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETriggerTowerTask");
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Et map SM%02d", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EETTT FineGrainVeto SM%02d", ism);
      me_i01_[ism-1] = mui_->collate3D(histo, histo, "EcalEndcap/Sums/EETriggerTowerTask");
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT FineGrainVeto SM%02d", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EETTT Flags SM%02d", ism);
      me_j01_[ism-1] = mui_->collate3D(histo, histo, "EcalEndcap/Sums/EETriggerTowerTask");
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Flags SM%02d", ism);
      mui_->add(me_j01_[ism-1], histo);

      for (int j = 0; j < 68 ; j++) {

        sprintf(histo, "EETTT Et T SM%02d TT%02d", ism, j+1);
        me_k01_[ism-1][j] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETriggerTowerTask/EnergyMaps");
        sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d", ism, j+1);
        mui_->add(me_k01_[ism-1][j], histo);

        sprintf(histo, "EETTT Et R SM%02d TT%02d", ism, j+1);
        me_k02_[ism-1][j] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETriggerTowerTask/EnergyMaps");
        sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d", ism, j+1);
        mui_->add(me_k02_[ism-1][j], histo);

      }

    }

  }

}

void EETriggerTowerClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Et map SM%02d", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT FineGrainVeto SM%02d", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Flags SM%02d", ism);
    mui_->subscribeNew(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribeNew(histo, ism);
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribeNew(histo, ism);
    }

  }

}

void EETriggerTowerClient::unsubscribe(void){

  if ( verbose_ ) cout << "EETriggerTowerClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETriggerTowerClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);

        mui_->removeCollate(me_j01_[ism-1]);

        for (int j = 0; j < 68 ; j++) {

          mui_->removeCollate(me_k01_[ism-1][j]);

          mui_->removeCollate(me_k02_[ism-1][j]);

        }

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Et map SM%02d", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT FineGrainVeto SM%02d", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EETTT Flags SM%02d", ism);
    mui_->unsubscribe(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
    }

  }

}

void EETriggerTowerClient::softReset(void){

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meh01_[ism-1] ) mui_->softReset(meh01_[ism-1]);
    if ( mei01_[ism-1] ) mui_->softReset(mei01_[ism-1]);
    if ( mej01_[ism-1] ) mui_->softReset(mej01_[ism-1]);

    for (int j = 0; j < 68 ; j++) {

      if ( mek01_[ism-1][j] ) mui_->softReset(mek01_[ism-1][j]);
      if ( mek02_[ism-1][j] ) mui_->softReset(mek02_[ism-1][j]);

    }

  }

}

void EETriggerTowerClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EETriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETriggerTowerTask/EETTT Et map SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EETTT Et map SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETriggerTowerTask/EETTT FineGrainVeto SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EETTT FineGrainVeto SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    i01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i01_[ism-1] );
    mei01_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETriggerTowerTask/EETTT Flags SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EETTT Flags SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    j01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j01_[ism-1] );
    mej01_[ism-1] = me;

    for (int j = 0; j < 68 ; j++) {

      if ( collateSources_ ) {
        sprintf(histo, "EcalEndcap/Sums/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d", ism, j+1);;
      } else {
        sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T SM%02d TT%02d").c_str(), ism, j+1);
      }
      me = mui_->get(histo);
      k01_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k01_[ism-1][j] );
      mek01_[ism-1][j] = me;

      if ( collateSources_ ) {
        sprintf(histo, "EcalEndcap/Sums/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d", ism, j+1);;
      } else {
        sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R SM%02d TT%02d").c_str(), ism, j+1);
      }
      me = mui_->get(histo);
      k02_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k02_[ism-1][j] );
      mek02_[ism-1][j] = me;

    }

  }

}

void EETriggerTowerClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EETriggerTowerClient html output ..." << std::endl;

  std::ofstream htmlFile[19];

  htmlFile[0].open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile[0] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile[0] << "<html>  " << std::endl;
  htmlFile[0] << "<head>  " << std::endl;
  htmlFile[0] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile[0] << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile[0] << "  <title>Monitor:TriggerTowerTask output</title> " << std::endl;
  htmlFile[0] << "</head>  " << std::endl;
  htmlFile[0] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile[0] << "<body>  " << std::endl;
  //htmlFile[0] << "<br>  " << std::endl;
  htmlFile[0] << "<a name=""top""></a>" << endl;
  htmlFile[0] << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile[0] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << std::endl;
  htmlFile[0] << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h2> " << std::endl;
  htmlFile[0] << "<br>" << std::endl;
  //htmlFile[0] << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  htmlFile[0] << "<hr>" << std::endl;
  htmlFile[0] << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile[0] << "<td bgcolor=white><a href=""#" << superModules_[i] << ">"
                << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile[0] << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  //const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy( "dummy", "dummy for sm", 17, 0., 17., 4, 0., 4. );
  for ( int i = 0; i < 68; i++ ) {
    dummy.Fill( i/4, i%4, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  string imgName, meName, imgMeName;

  TCanvas* cMe1 = new TCanvas("cMe1", "Temp", 2*csize, csize);
//  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.2*csize), int(0.4*csize));
//  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.4*csize), int(0.4*csize));
  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.8*csize), int(0.9*csize));
  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.9*csize), int(0.9*csize));

  TProfile2D* objp;
  TH2F* obj2f;
  TH3F* obj3f;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    if ( i>0 ) htmlFile[0] << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;
    htmlFile[0] << "<h3><a name=""" << ism << """></a><strong>Supermodule&nbsp;&nbsp;"
                << ism << "</strong></h3>" << endl;

    // ---------------------------  Et plot

    imgName = "";

    objp = h01_[ism-1];

    if ( objp ) {

      meName = objp->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgName = meName + ".png";
      imgMeName = htmlDir + imgName;

      cMe1->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objp->GetXaxis()->SetNdivisions(17);
      objp->GetYaxis()->SetNdivisions(4);
      cMe1->SetGridx();
      cMe1->SetGridy();
      objp->Draw("colz");
      dummy.Draw("text,same");
      cMe1->Update();
      cMe1->SaveAs(imgMeName.c_str());

    }

    htmlFile[0] << "<img src=\"" << imgName << "\"><br>" << std::endl;

    std::stringstream subpage;
    subpage << htmlName.substr( 0, htmlName.find( ".html" ) ) << "_SM" << ism << ".html" << std::ends;
    htmlFile[0] << "<a href=\"" << subpage.str().c_str() << "\">SM" << ism << " details</a><br>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;

    htmlFile[ism].open((htmlDir + subpage.str()).c_str());

    // html page header
    htmlFile[ism] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
    htmlFile[ism] << "<html>  " << std::endl;
    htmlFile[ism] << "<head>  " << std::endl;
    htmlFile[ism] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
    htmlFile[ism] << " http-equiv=\"content-type\">  " << std::endl;
    htmlFile[ism] << "  <title>Monitor:TriggerTowerTask output SM" << ism << "</title> " << std::endl;
    htmlFile[ism] << "</head>  " << std::endl;
    htmlFile[ism] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
    htmlFile[ism] << "<body>  " << std::endl;
    htmlFile[ism] << "<br>  " << std::endl;
    htmlFile[ism] << "<h3>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h3>" << std::endl;
    htmlFile[ism] << "<h3>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h3> " << std::endl;
    htmlFile[ism] << "<h3>SM:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << ism << "</span></h3>" << std::endl;
    htmlFile[ism] << "<hr>" << std::endl;

    // ---------------------------  Flag bits plots

    htmlFile[ism] << "<h3><strong>Trigger Tower Flags</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    obj3f = j01_[ism-1];

    if ( obj3f ) {

      imgName = "";

      meName = obj3f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }

      int counter = 0;

      for ( int j=1; j<8; j++ ) {

        if ( j == 3 ) continue;   //  010 bits combination is not used
        counter++;
        if ( j < 7 ) {
          imgName = meName + "_" + char(47+j) + ".png";
          obj3f->GetZaxis()->SetRange( j, j );
        }
        else {
          imgName = meName + "_6-7.png";
          obj3f->GetZaxis()->SetRange( j, j+1 );
        }
        imgMeName = htmlDir + imgName;

        obj2f = (TH2F*) obj3f->Project3D( "yx" );

        cMe2->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cMe2->SetGridx();
        cMe2->SetGridy();

        std::stringstream title;
        if ( j < 7 ) {
          title << "EETTT Flags SM" << std::setfill('0') << std::setw(2) << ism << ", bit " << bitset<3>(j-1);
        } else {
          title << "EETTT Flags SM" << std::setfill('0') << std::setw(2) << ism << " bits 110+111";
        }
        obj2f->SetTitle( title.str().c_str() );

        obj2f->Draw("colz");
        dummy.Draw("text,same");
        cMe2->Update();
        cMe2->SaveAs(imgMeName.c_str());

        htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;

        if ( counter%2 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;

      }

    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // ---------------------------  Fine Grain Veto

    htmlFile[ism] << "<h3><strong>Fine Grain Veto</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    obj3f = i01_[ism-1];

    if ( obj3f ) {

      imgName = "";

      meName = obj3f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }

      for ( int j=1; j<=2; j++ ) {

        imgName = meName + "_" + char(47+j) + ".png";
        imgMeName = htmlDir + imgName;

        obj3f->GetZaxis()->SetRange( j, j );

        obj2f = (TH2F*) obj3f->Project3D( "yx" );

        cMe2->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cMe2->SetGridx();
        cMe2->SetGridy();

        std::stringstream title;
        title << "EETTT FineGrainVeto SM" << std::setfill('0') << std::setw(2) << ism << ", FineGrainVeto = " << j-1;
        obj2f->SetTitle( title.str().c_str() );

        obj2f->Draw("colz");
        dummy.Draw("text,same");
        cMe2->Update();
        cMe2->SaveAs(imgMeName.c_str());

        htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;

      }

    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;


    // ---------------------------  Et plots per Tower

    htmlFile[ism] << "<h3><strong>Et</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    for ( int j=0; j<68; j++ ) {

      TH1F* obj1f1 = k01_[ism-1][j];
      TH1F* obj1f2 = k02_[ism-1][j];

      if ( obj1f1 ) {

        imgName = "";

        meName = obj1f1->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }

        imgName = meName + ".png";
        imgMeName = htmlDir + imgName;

        cMe3->cd();
        gStyle->SetOptStat("euomr");
        if ( obj1f2 ) {
          float m = TMath::Max( obj1f1->GetMaximum(), obj1f2->GetMaximum() );
          obj1f1->SetMaximum( m + 1. );
        }
        obj1f1->SetStats(kTRUE);
        gStyle->SetStatW( gStyle->GetStatW() * 1.5 );
        obj1f1->Draw();
        cMe3->Update();

        if ( obj1f2 ) {
          gStyle->SetStatY( gStyle->GetStatY() - 1.25*gStyle->GetStatH() );
          gStyle->SetStatTextColor( kRed );
          obj1f2->SetStats(kTRUE);
          obj1f2->SetLineColor( kRed );
          obj1f2->Draw( "sames" );
          cMe3->Update();
          gStyle->SetStatY( gStyle->GetStatY() + 1.25*gStyle->GetStatH() );
          gStyle->SetStatTextColor( kBlack );
        }

        gStyle->SetStatW( gStyle->GetStatW() / 1.5 );
        cMe3->SaveAs(imgMeName.c_str());

        htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;

      }

      if ( (j+1)%4 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;

    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // html page footer
    htmlFile[ism] << "</body> " << std::endl;
    htmlFile[ism] << "</html> " << std::endl;
    htmlFile[ism].close();
  }

  delete cMe1;
  delete cMe2;
  delete cMe3;

  // html page footer
  htmlFile[0] << "</body> " << std::endl;
  htmlFile[0] << "</html> " << std::endl;
  htmlFile[0].close();

}

