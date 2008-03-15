/*
 * \file EEBeamHodoClient.cc
 *
 * $Date: 2008/03/14 14:38:58 $
 * $Revision: 1.25 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "TCanvas.h"
#include "TStyle.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEBeamHodoClient::EEBeamHodoClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  hm01_    = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

EEBeamHodoClient::~EEBeamHodoClient(){

}

void EEBeamHodoClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( verbose_ ) cout << "EEBeamHodoClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEBeamHodoClient::beginRun(void){

  if ( verbose_ ) cout << "EEBeamHodoClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEBeamHodoClient::endJob(void) {

  if ( verbose_ ) cout << "EEBeamHodoClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  if ( cloneME_ ) {

    for (int i=0; i<4; i++) {

      if ( ho01_[i] ) delete ho01_[i];
      if ( hr01_[i] ) delete hr01_[i];

    }

    if ( hp01_[0] ) delete hp01_[0];
    if ( hp01_[1] ) delete hp01_[1];

    if ( hp02_ ) delete hp02_;

    if ( hs01_[0] ) delete hs01_[0];
    if ( hs01_[1] ) delete hs01_[1];

    if ( hq01_[0] ) delete hq01_[0];
    if ( hq01_[1] ) delete hq01_[1];

    if ( ht01_ ) delete ht01_;

    if ( hc01_[0] ) delete hc01_[0];
    if ( hc01_[1] ) delete hc01_[1];
    if ( hc01_[2] ) delete hc01_[2];

    if ( hm01_ )    delete hm01_;

    if ( he01_[0] ) delete he01_[0];
    if ( he01_[1] ) delete he01_[1];

    if ( he02_[0] ) delete he02_[0];
    if ( he02_[1] ) delete he02_[1];

    if ( he03_[0] ) delete he03_[0];
    if ( he03_[1] ) delete he03_[1];
    if ( he03_[2] ) delete he03_[2];

  }

  for (int i=0; i<4; i++) {

    ho01_[i] = 0;
    hr01_[i] = 0;

  }

  hp01_[0] = 0;
  hp01_[1] = 0;

  hp02_ = 0;

  hs01_[0] = 0;
  hs01_[1] = 0;

  hq01_[0] = 0;
  hq01_[1] = 0;

  ht01_ = 0;

  hc01_[0] = 0;
  hc01_[1] = 0;
  hc01_[2] = 0;

  hm01_    = 0;

  he01_[0] = 0;
  he01_[1] = 0;

  he02_[0] = 0;
  he02_[1] = 0;

  he03_[0] = 0;
  he03_[1] = 0;
  he03_[2] = 0;

}

void EEBeamHodoClient::endRun(void) {

  if ( verbose_ ) cout << "EEBeamHodoClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEBeamHodoClient::setup(void) {

  dbe_->setCurrentFolder( "EcalEndcap/EEBeamHodoClient" );

}

void EEBeamHodoClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  dbe_->setCurrentFolder( "EcalEndcap/EEBeamHodoClient" );

}

bool EEBeamHodoClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EEBeamHodoClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EEBeamHodoClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  int smId = 1;

  char histo[200];

  MonitorElement* me;

  for (int i=0; i<4; i++) {

    sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT occup %s %02d").c_str(), Numbers::sEE(smId).c_str(), i+1);
    me = dbe_->get(histo);
    ho01_[i] = UtilsClient::getHisto<TH1F*>( me, cloneME_, ho01_[i] );

    sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT raw %s %02d").c_str(), Numbers::sEE(smId).c_str(), i+1);
    me = dbe_->get(histo);
    hr01_[i] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hr01_[i] );

  }

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT PosX rec %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hp01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT PosY rec %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hp01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hp01_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT PosYX rec %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hp02_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hp02_ );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT SloX %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hs01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT SloY %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hs01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hs01_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT QualX %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hq01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT QualY %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hq01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hq01_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT TDC rec %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  ht01_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, ht01_ );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT Hodo-Calo X vs Cry %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hc01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT Hodo-Calo Y vs Cry %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hc01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT TDC-Calo vs Cry %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hc01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, hc01_[2] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT Missing Collections %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  hm01_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hm01_ );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT prof E1 vs X %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he01_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT prof E1 vs Y %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he01_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, he01_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT his E1 vs X %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he02_[0] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he02_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT his E1 vs Y %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he02_[1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, he02_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT PosX Hodo-Calo %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he03_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[0] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT PosY Hodo-Calo %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he03_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[1] );

  sprintf(histo, (prefixME_+"EcalEndcap/EEBeamHodoTask/EEBHT TimeMax TDC-Calo %s").c_str(), Numbers::sEE(smId).c_str());
  me = dbe_->get(histo);
  he03_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, he03_[2] );

}

void EEBeamHodoClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EEBeamHodoClient html output ..." << endl;

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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">BeamHodo</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile <<  "<a href=\"#Hodo_raw\"> Hodoscope raw </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#Hodo_reco\"> Hodoscope reco </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#Hodo-Calo\"> Hodo-Calo </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#eneVspos\"> Energy vs position </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#missingColl\"> Missing collections </a>" << endl;
  htmlFile << "<p>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<p>" << endl;

  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"Hodo_raw\"> <B> Hodoscope raw plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;


  const int csize = 250;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for ( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  string imgNameP, imgNameR, imgName, meName;

  TCanvas* cP = new TCanvas("cP", "Temp", csize, csize);

  TH1F* obj1f;
  TH2F* obj2f;
  TProfile* objp;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int i=0; i<4; i++) {

    imgNameP = "";

    obj1f = ho01_[i];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      gPad->SetLogy(0);
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    imgNameR = "";

    obj1f = hr01_[i];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameR = meName + ".png";
      imgName = htmlDir + imgNameR;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameR.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameR << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( i == 1 ) {
      htmlFile << "</tr>" << endl;
      htmlFile << "<tr align=\"center\">" << endl;
    }

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"Hodo_reco\"> <B> Hodoscope reco plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;


  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int i=0; i<2; i++) {

    imgNameP = "";

    obj1f = hp01_[i];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  obj2f = hp02_;

  imgNameP = "";

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameP = meName + ".png";
    imgName = htmlDir + imgNameP;

    cP->cd();
//    gStyle->SetOptStat("euomr");
//    obj2f->SetStats(kTRUE);
    obj2f->Draw("");
    cP->Update();
    cP->SaveAs(imgName.c_str());

  }

  if ( imgNameP.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int i=0; i<4; i++) {

    obj1f = 0;

    imgNameP = "";

    switch ( i ) {
    case 0:
      obj1f = hs01_[0];
      break;
    case 1:
      obj1f = hs01_[1];
      break;
    case 2:
      obj1f = hq01_[0];
      break;
    case 3:
      obj1f = hq01_[1];
      break;
    default:
      break;
    }

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"Hodo-Calo\"> <B> Hodo-Calo plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;


  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int i=0; i<3; i++) {

    obj1f = 0;

    imgNameP = "";

    switch ( i ) {
    case 0:
      obj1f = hc01_[0];
      break;
    case 1:
      obj1f = hc01_[1];
      break;
    case 2:
      obj1f = hc01_[2];
      break;
    default:
      break;
    }

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      gPad->SetLogy(0);
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;


  for (int i=0; i<3; i++) {

    obj1f = 0;

    imgNameP = "";

    switch ( i ) {
    case 0:
      obj1f = he03_[0];
      break;
    case 1:
      obj1f = he03_[1];
      break;
    case 2:
      obj1f = he03_[2];
      break;
    default:
      break;
    }

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
//      if ( obj1f->GetMaximum(histMax) > 0. ) {
//        gPad->SetLogy(1);
//      } else {
//        gPad->SetLogy(0);
//      }
      obj1f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());

      gPad->SetLogy(0);

    }


    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"eneVspos\"> <B> Energy vs position plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;


  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int i=0; i<2; i++) {

    objp = 0;

    imgNameP = "";

    switch ( i ) {
    case 0:
      objp = he01_[0];
      break;
    case 1:
      objp = he01_[1];
      break;
    default:
      break;
    }

    if ( objp ) {

      meName = objp->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameP = meName + ".png";
      imgName = htmlDir + imgNameP;

      cP->cd();
      gStyle->SetOptStat("euomr");
      objp->SetStats(kTRUE);
      objp->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    obj2f = 0;

    imgNameR = "";

    switch ( i ) {
    case 0:
      obj2f = he02_[0];
    break;
    case 1:
      obj2f = he02_[1];
      break;
    default:
      break;
    }

    if ( obj2f ) {

      meName = obj2f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameR = meName + ".png";
      imgName = htmlDir + imgNameR;

      cP->cd();
//      gStyle->SetOptStat("euomr");
//      obj2f->SetStats(kTRUE);
      obj2f->Draw();
      cP->Update();
      cP->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if ( imgNameP.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameR.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameR << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"missingColl\"> <B> Missing collections  </B> </a> " << endl;
  htmlFile << "</br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  obj1f = hm01_;

  imgNameP = "";

  if ( obj1f ) {

    meName = obj1f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameP = meName + ".png";
    imgName = htmlDir + imgNameP;

    cP->cd();
    gStyle->SetOptStat("euomr");
    obj1f->SetStats(kTRUE);
    obj1f->Draw();
    cP->Update();
    cP->SaveAs(imgName.c_str());
    gPad->SetLogy(0);

  }

  if ( imgNameP.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameP << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cP;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile.close();

}

