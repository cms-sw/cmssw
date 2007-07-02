/*
 * \file EBTriggerTowerClient.cc
 *
 * $Date: 2007/02/01 15:25:25 $
 * $Revision: 1.25 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBTriggerTowerClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBTriggerTowerClient::EBTriggerTowerClient(const ParameterSet& ps){

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

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i < 37; i++ ) superModules_.push_back(i);
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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meh01_[ism-1] = 0;

    mei01_[ism-1] = 0;

    mej01_[ism-1] = 0;

    for ( int j = 0; j < 68 ; j++ ) {

      mek01_[ism-1][j] = 0;
      mek02_[ism-1][j] = 0;

    }

  }

}

EBTriggerTowerClient::~EBTriggerTowerClient(){

}

void EBTriggerTowerClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EBTriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTriggerTowerClient::beginRun(void){

  if ( verbose_ ) cout << "EBTriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBTriggerTowerClient::endJob(void) {

  if ( verbose_ ) cout << "EBTriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBTriggerTowerClient::endRun(void) {

  if ( verbose_ ) cout << "EBTriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBTriggerTowerClient::setup(void) {

//  Char_t histo[200];
//
//  mui_->setCurrentFolder( "EcalBarrel/EBTriggerTowerClient" );
//  DaqMonitorBEInterface* bei = mui_->getBEInterface();
//
//  for ( unsigned int i=0; i<superModules_.size(); i++ ) {
//
//    int ism = superModules_[i];
//
//  }
//
//  for ( unsigned int i=0; i<superModules_.size(); i++ ) {
//
//    int ism = superModules_[i];
//
//    for ( int ie = 1; ie <= 85; ie++ ) {
//      for ( int ip = 1; ip <= 20; ip++ ) {
//
//      }
//    }
//
//  }

}

void EBTriggerTowerClient::cleanup(void) {

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

//  mui_->setCurrentFolder( "EcalBarrel/EBTriggerTowerClient" );
//  DaqMonitorBEInterface* bei = mui_->getBEInterface();
//
//  for ( unsigned int i=0; i<superModules_.size(); i++ ) {
//
//    int ism = superModules_[i];
//
//  }

}

bool EBTriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism) {

  bool status = true;

  return status;

}

void EBTriggerTowerClient::subscribe(void){

  if ( verbose_ ) cout << "EBTriggerTowerClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Et map SM%02d", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d", ism);
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Flags SM%02d", ism);
    mui_->subscribe(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
    }

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBTriggerTowerClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EBTTT Et map SM%02d", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTriggerTowerTask");
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Et map SM%02d", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBTTT FineGrainVeto SM%02d", ism);
      me_i01_[ism-1] = mui_->collate3D(histo, histo, "EcalBarrel/Sums/EBTriggerTowerTask");
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EBTTT Flags SM%02d", ism);
      me_j01_[ism-1] = mui_->collate3D(histo, histo, "EcalBarrel/Sums/EBTriggerTowerTask");
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Flags SM%02d", ism);
      mui_->add(me_j01_[ism-1], histo);

      for (int j = 0; j < 68 ; j++) {

        sprintf(histo, "EBTTT Et T SM%02d TT%02d", ism, j+1);
        me_k01_[ism-1][j] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTriggerTowerTask/EnergyMaps");
        sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d", ism, j+1);
        mui_->add(me_k01_[ism-1][j], histo);

        sprintf(histo, "EBTTT Et R SM%02d TT%02d", ism, j+1);
        me_k02_[ism-1][j] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTriggerTowerTask/EnergyMaps");
        sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d", ism, j+1);
        mui_->add(me_k02_[ism-1][j], histo);

      }

    }

  }

}

void EBTriggerTowerClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Et map SM%02d", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d", ism);
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Flags SM%02d", ism);
    mui_->subscribeNew(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribeNew(histo, ism);
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribeNew(histo, ism);
    }

  }

}

void EBTriggerTowerClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBTriggerTowerClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBTriggerTowerClient: uncollate" << endl;

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

    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Et map SM%02d", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d", ism);
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EBTTT Flags SM%02d", ism);
    mui_->unsubscribe(histo, ism);

    for (int j = 0; j < 68 ; j++) {
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d", ism, j+1);
      mui_->subscribe(histo, ism);
    }

  }

}

void EBTriggerTowerClient::softReset(void){

}

void EBTriggerTowerClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBTriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTriggerTowerTask/EBTTT Et map SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EBTTT Et map SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    h01_[ism-1] = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EBTTT FineGrainVeto SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    i01_[ism-1] = EBMUtilsClient::getHisto<TH3F*>( me, cloneME_, i01_[ism-1] );
    mei01_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTriggerTowerTask/EBTTT Flags SM%02d", ism);
    } else {
      sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EBTTT Flags SM%02d").c_str(), ism);
    }
    me = mui_->get(histo);
    j01_[ism-1] = EBMUtilsClient::getHisto<TH3F*>( me, cloneME_, j01_[ism-1] );
    mej01_[ism-1] = me;

    for (int j = 0; j < 68 ; j++) {

      if ( collateSources_ ) {
        sprintf(histo, "EcalBarrel/Sums/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d", ism, j+1);;
      } else {
        sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et T SM%02d TT%02d").c_str(), ism, j+1);
      }
      me = mui_->get(histo);
      k01_[ism-1][j] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, k01_[ism-1][j] );
      mek01_[ism-1][j] = me;

      if ( collateSources_ ) {
        sprintf(histo, "EcalBarrel/Sums/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d", ism, j+1);;
      } else {
        sprintf(histo, (prefixME_+"EcalBarrel/EBTriggerTowerTask/EnergyMaps/EBTTT Et R SM%02d TT%02d").c_str(), ism, j+1);
      }
      me = mui_->get(histo);
      k02_[ism-1][j] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, k02_[ism-1][j] );
      mek02_[ism-1][j] = me;

    }

  }

}

std::string binary( int i ) {
  if( i == 0 ) return( "0" );
  std::string s;
  while( i > 0 ) {
    s = char( 48 + (i&1) ) + s;
    i = i >> 1;
  }
  return s;
}

void EBTriggerTowerClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBTriggerTowerClient html output ..." << std::endl;

  std::ofstream htmlFile[37];

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
  htmlFile[0] << "<br>  " << std::endl;
  htmlFile[0] << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile[0] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << std::endl;
  htmlFile[0] << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h2> " << std::endl;
  htmlFile[0] << "<hr>" << std::endl;
  //htmlFile[0] << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  //htmlFile[0] << "<hr>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  //const double histMax = 1.e15;

  //int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  TH2C dummy( "dummy", "dummy for sm", 17, 0., 17., 4, 0., 4. );
  for ( int i = 0; i < 68; i++ ) {
    dummy.Fill( i/4, i%4, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  string imgName, meName, imgFullName;

  TCanvas* rectangle = new TCanvas("rectangle", "Temp", 2*csize, csize);
  TCanvas* rectsmall = new TCanvas("rectangle small", "Temp", int(1.6*csize), int(0.8*csize));
  TCanvas* square    = new TCanvas("square small", "Temp", int(0.8*csize), int(0.8*csize));

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    htmlFile[0] << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << std::endl;

////  --------> no quality plot yet... 
//     // Quality plot

//     imgName = "";
    TH2F* obj2f;
//     obj2f = EBMUtilsClient::getHisto<TH2F*>( meg???_[ism-1] );
//     if ( obj2f ) {
//       meName = obj2f->GetName();
//       for ( unsigned int i = 0; i < meName.size(); i++ ) {
//         if ( meName.substr(i, 1) == " " )  {
//           meName.replace(i, 1, "_");
//         }
//       }
//       imgName = meName + ".png";
//       imgFullName = htmlDir + imgName;
//       rectangle->cd();
//       gStyle->SetOptStat(" ");
//       gStyle->SetPalette(6, pCol3);
//       obj2f->GetXaxis()->SetNdivisions(17);
//       obj2f->GetYaxis()->SetNdivisions(4);
//       rectangle->SetGridx();
//       rectangle->SetGridy();
//       obj2f->SetMinimum(-0.00000001);
//       obj2f->SetMaximum(6.0);
//       obj2f->Draw("col");
//       dummy.Draw("text,same");
//       rectangle->Update();
//       rectangle->SaveAs(imgName.c_str());
//     }

    
    // ---------------------------  Et plot

    imgName = "";

    TProfile2D* objp = h01_[ism-1];
    if ( objp ) {
      meName = objp->GetName();
      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgName = meName + ".png";
      imgFullName = htmlDir + imgName;
      rectangle->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette( 1 );
      objp->GetXaxis()->SetNdivisions(17);
      objp->GetYaxis()->SetNdivisions(4);
      rectangle->SetGridx();
      rectangle->SetGridy();
      objp->SetMinimum(0.00000001);
      objp->Draw("colz");
      dummy.Draw("text,same");
      rectangle->Update();
      rectangle->SaveAs(imgFullName.c_str());
    }

    htmlFile[0] << "<img src=\"" << imgName << "\"><br>" << std::endl;

    std::stringstream subpage;
    subpage << htmlName.substr( 0, htmlName.find( ".html" ) ) << "_SM" << ism << ".html" << std::ends;
    htmlFile[0] << "<a href=" << subpage.str() << ">SM" << ism << " details</a><br>" << std::endl; 
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

    TH3F* obj3f = j01_[ism-1];
    if ( obj3f ) {
      imgName = "";
      meName = obj3f->GetName();
      for ( unsigned int i = 0; i < meName.size(); i++ ) {
	if ( meName.substr(i, 1) == " " )  {
	  meName.replace(i, 1 ,"_" );
	}
      }
      int counter = 0;
      for( int j=1; j<=7; j++ ) {
	if( j == 3 ) continue;   //  010 bits combination is not used 
	counter++;
	if( j <= 6 ) {
	  imgName = meName + "_" + char(48+j) + ".png";
	}
	else {
	  imgName = meName + "_6-7.png";
	}
	imgFullName = htmlDir + imgName;
	
	if( j != 6 ) {
	  obj3f->GetZaxis()->SetRange( j, j );
	}
	else {
	  obj3f->GetZaxis()->SetRange( j, j );    
	}
	obj2f = (TH2F*) obj3f->Project3D( "yx" );
	rectsmall->cd();
	gStyle->SetOptStat(" ");
	gStyle->SetPalette( 1 );
	obj2f->GetXaxis()->SetNdivisions(17);
	obj2f->GetYaxis()->SetNdivisions(4);
	rectsmall->SetGridx();
	rectsmall->SetGridy();
	obj2f->SetMinimum(0.00000001);
	std::stringstream title; 
	if( j <= 6 ) { title << "EBTTT Flags SM" << ism << ", bit " << binary(j-1); }
	else         { title << "EBTTT Flags SM" << ism << " bits 110+111"; }
	obj2f->SetTitle( title.str().c_str() );
	obj2f->Draw("colz");
	dummy.Draw("text,same");
	rectsmall->Update();
	rectsmall->SaveAs(imgFullName.c_str()); 
	htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;
	if( counter%2 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl; 
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
      for( int j=1; j<=2; j++ ) {
	imgName = meName + char(48+j) + ".png";
	imgFullName = htmlDir + imgName;
	obj3f->GetZaxis()->SetRange( j, j );
	obj2f = (TH2F*) obj3f->Project3D( "yx" );
	rectsmall->cd();
	gStyle->SetOptStat(" ");
	gStyle->SetPalette( 1 );
	obj2f->GetXaxis()->SetNdivisions(17);
	obj2f->GetYaxis()->SetNdivisions(4);
	rectsmall->SetGridx();
	rectsmall->SetGridy();
	obj2f->SetMinimum(0.00000001);
	std::stringstream title; 
	title << "EBTTT FineGrainVeto SM" << ism << ", FineGrainVeto = " << j-1;
	obj2f->SetTitle( title.str().c_str() );
	obj2f->Draw("colz");
	dummy.Draw("text,same");
	rectsmall->Update();
	rectsmall->SaveAs(imgFullName.c_str()); 
	htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;
      }
    }
    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;


    // ---------------------------  Et plots per Tower
      
    htmlFile[ism] << "<h3><strong>Et</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;
    

    for( int j=0; j<68; j++ ) {
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
	imgFullName = htmlDir + imgName;
	square->cd();
	gStyle->SetOptStat("euomr");
	if( obj1f2 ) {
	  float m = TMath::Max( obj1f1->GetMaximum(), obj1f2->GetMaximum() );
	  obj1f1->SetMaximum( m + 1. );
	}
	obj1f1->SetStats(kTRUE);
	gStyle->SetStatW( gStyle->GetStatW() * 1.5 );
	obj1f1->Draw();
	square->Update();
	if( obj1f2 ) {
	  gStyle->SetStatY( gStyle->GetStatY() - 1.25*gStyle->GetStatH() );
	  gStyle->SetStatTextColor( kRed );
	  obj1f2->SetStats(kTRUE);
	  obj1f2->SetLineColor( kRed );
	  obj1f2->Draw( "sames" );
	  square->Update();
	  gStyle->SetStatY( gStyle->GetStatY() + 1.25*gStyle->GetStatH() );
	  gStyle->SetStatTextColor( kBlack );
	}
	gStyle->SetStatW( gStyle->GetStatW() / 1.5 );
	square->SaveAs(imgFullName.c_str()); 
	htmlFile[ism] << "<td><img src=\"" << imgName << "\"></td>" << std::endl;
      }
      if( (j+1)%4 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;
    }
    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // html page footer
    htmlFile[ism] << "</body> " << std::endl;
    htmlFile[ism] << "</html> " << std::endl;
    htmlFile[ism].close();
  }

  delete rectangle;
  delete rectsmall;
  delete square;

  // html page footer
  htmlFile[0] << "</body> " << std::endl;
  htmlFile[0] << "</html> " << std::endl;
  htmlFile[0].close();

}

