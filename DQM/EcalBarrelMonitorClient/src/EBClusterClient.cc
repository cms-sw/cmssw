/*
 * \file EBClusterClient.cc
 *
 * $Date: 2008/03/15 14:07:44 $
 * $Revision: 1.58 $
 * \author G. Della Ricca
 * \author F. Cossutti
 * \author E. Di Marco
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGaxis.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBClusterClient::EBClusterClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  h01_[0] = 0;
  h01_[1] = 0;
  h01_[2] = 0;

  h02_[0] = 0;
  h02ProjEta_[0] = 0;
  h02ProjPhi_[0] = 0;
  h02_[1] = 0;
  h02ProjEta_[1] = 0;
  h02ProjPhi_[1] = 0;

  h03_ = 0;
  h03ProjEta_ = 0;
  h03ProjPhi_ = 0;

  h04_ = 0;
  h04ProjEta_ = 0;
  h04ProjPhi_ = 0;

  i01_[0] = 0;
  i01_[1] = 0;
  i01_[2] = 0;

  s01_[0] = 0;
  s01_[1] = 0;
  s01_[2] = 0;

}

EBClusterClient::~EBClusterClient(){

}

void EBClusterClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( verbose_ ) cout << "EBClusterClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBClusterClient::beginRun(void){

  if ( verbose_ ) cout << "EBClusterClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBClusterClient::endJob(void) {

  if ( verbose_ ) cout << "EBClusterClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBClusterClient::endRun(void) {

  if ( verbose_ ) cout << "EBClusterClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBClusterClient::setup(void) {

  dbe_->setCurrentFolder( "EcalBarrel/EBClusterClient" );

}

void EBClusterClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {
    if ( h01_[0] ) delete h01_[0];
    if ( h01_[1] ) delete h01_[1];
    if ( h01_[2] ) delete h01_[2];

    if ( h02_[0] ) delete h02_[0];
    if ( h02ProjEta_[0] ) delete h02ProjEta_[0];
    if ( h02ProjPhi_[0] ) delete h02ProjPhi_[0];
    if ( h02_[1] ) delete h02_[1];
    if ( h02ProjEta_[1] ) delete h02ProjEta_[1];
    if ( h02ProjPhi_[1] ) delete h02ProjPhi_[1];

    if ( h03_ ) delete h03_;
    if ( h03ProjEta_ ) delete h03ProjEta_;
    if ( h03ProjPhi_ ) delete h03ProjPhi_;
    if ( h04_ ) delete h04_;
    if ( h04ProjEta_ ) delete h04ProjEta_;
    if ( h04ProjPhi_ ) delete h04ProjPhi_;

    if ( i01_[0] ) delete i01_[0];
    if ( i01_[1] ) delete i01_[1];
    if ( i01_[2] ) delete i01_[2];

    if ( s01_[0] ) delete s01_[0];
    if ( s01_[1] ) delete s01_[1];
    if ( s01_[2] ) delete s01_[2];

  }

  h01_[0] = 0;
  h01_[1] = 0;
  h01_[2] = 0;

  h02_[0] = 0;
  h02ProjEta_[0] = 0;
  h02ProjPhi_[0] = 0;
  h02_[1] = 0;
  h02ProjEta_[1] = 0;
  h02ProjPhi_[1] = 0;

  h03_ = 0;
  h03ProjEta_ = 0;
  h03ProjPhi_ = 0;
  h04_ = 0;
  h04ProjEta_ = 0;
  h04ProjPhi_ = 0;

  i01_[0] = 0;
  i01_[1] = 0;
  i01_[2] = 0;

  s01_[0] = 0;
  s01_[1] = 0;
  s01_[2] = 0;

}

bool EBClusterClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EBClusterClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBClusterClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC energy");
  me = dbe_->get(histo);
  h01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC size");
  me = dbe_->get(histo);
  h01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC number");
  me = dbe_->get(histo);
  h01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[2] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC energy map");
  me = dbe_->get(histo);
  h02_[0] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC ET map");
  me = dbe_->get(histo);
  h02_[1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC number map");
  me = dbe_->get(histo);
  h03_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, h03_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC size map");
  me = dbe_->get(histo);
  h04_ = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC energy projection eta");
  me = dbe_->get(histo);
  h02ProjEta_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjEta_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC energy projection phi");
  me = dbe_->get(histo);
  h02ProjPhi_[0] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjPhi_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC ET projection eta");
  me = dbe_->get(histo);
  h02ProjEta_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjEta_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC ET projection phi");
  me = dbe_->get(histo);
  h02ProjPhi_[1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, h02ProjPhi_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC number projection eta");
  me = dbe_->get(histo);
  h03ProjEta_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03ProjEta_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC number projection phi");
  me = dbe_->get(histo);
  h03ProjPhi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03ProjPhi_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC size projection eta");
  me = dbe_->get(histo);
  h04ProjEta_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, h04ProjEta_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT BC size projection phi");
  me = dbe_->get(histo);
  h04ProjPhi_ = UtilsClient::getHisto<TProfile*>( me, cloneME_, h04ProjPhi_ );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT SC energy");
  me = dbe_->get(histo);
  i01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT SC size");
  me = dbe_->get(histo);
  i01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT SC number");
  me = dbe_->get(histo);
  i01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[2] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT s1s9");
  me = dbe_->get(histo);
  s01_[0] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[0] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT s9s25");
  me = dbe_->get(histo);
  s01_[1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[1] );

  sprintf(histo, "EcalBarrel/EBClusterTask/EBCLT dicluster invariant mass");
  me = dbe_->get(histo);
  s01_[2] = UtilsClient::getHisto<TH1F*>( me, cloneME_, s01_[2] );

}

void EBClusterClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EBClusterClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:ClusterTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">CLUSTER</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  htmlFile <<  "<a href=\"#bc_plots\"> Basic Clusters plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#sc_plots\"> Super Clusters plots </a>" << endl;
  htmlFile << "<p>" << endl;
  htmlFile <<  "<a href=\"#hl_plots\"> Higher Level Quantities plots </a>" << endl;
  htmlFile << "<p>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<p>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize1D = 250;
  const int csize2D = 300;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  // dummy histogram labelling the SM's
  TH2C labelGrid("labelGrid","label grid for SM", 18, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 2, -1.479, 1.479);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 2 - sm/18;
    int z = x + 8;
    if ( z > 18 ) z = z - 18;
    if ( y == 1 ) {
      labelGrid.SetBinContent(x, y, -z);
    } else {
      labelGrid.SetBinContent(x, y, +z);
    }
  }
  labelGrid.SetMarkerSize(2);
  labelGrid.SetMinimum(-18.01);

  TGaxis Xaxis(-M_PI*(9+1.5)/9, -1.479, M_PI*(9-1.5)/9, -1.479, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 40306, "N");

  string imgNameB[3], imgNameBMap[4], imgNameS[3];
  string imgNameBXproj[4], imgNameBYproj[4];
  string imgNameHL[3], imgName, meName;

  TCanvas* cEne = new TCanvas("cEne", "Temp", csize1D, csize1D);
  TCanvas* cMap = new TCanvas("cMap", "Temp", int(360./170.*csize2D), csize2D);

  TH1F* obj1f;
  TProfile2D* objp;
  TH2F* obj2f;
  TProfile* obj1pX;
  TProfile* obj1pY;
  TH1F* obj1fX;
  TH1F* obj1fY;

  gStyle->SetPaintTextFormat("+g");

  // ==========================================================================
  // basic clusters
  // ==========================================================================

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameB[iCanvas-1] = "";

    obj1f = h01_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameB[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameB[iCanvas-1];

      cEne->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
    }
  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"bc_plots\"> <B> Basic Clusters plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    if ( imgNameB[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameB[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameBMap[iCanvas-1] = "";

    objp = (iCanvas!=3) ? h02_[iCanvas-1] : h04_;

    if ( objp ) {

      meName = objp->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameBMap[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameBMap[iCanvas-1];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      objp->GetXaxis()->SetNdivisions( 40118, kFALSE);
      objp->GetYaxis()->SetNdivisions(170102, kFALSE);
      cMap->SetGridx();
      cMap->SetGridy();
      objp->Draw("colz");
      labelGrid.Draw("text,same");
      Xaxis.Draw();
      cMap->Update();
      objp->GetXaxis()->SetLabelColor(0);
      cMap->SaveAs(imgName.c_str());
      objp->GetXaxis()->SetLabelColor(1);

      char projXName[100];
      char projYName[100];
      sprintf(projXName,"%s_px",meName.c_str());
      imgNameBXproj[iCanvas-1] = string(projXName) + ".png";
      sprintf(projYName,"%s_py",meName.c_str());
      imgNameBYproj[iCanvas-1] = string(projYName) + ".png";

      obj1pX = (iCanvas!=3) ? h02ProjEta_[iCanvas-1] : h04ProjEta_;
      obj1pY = (iCanvas!=3) ? h02ProjPhi_[iCanvas-1] : h04ProjPhi_;

      if (obj1pX && obj1pY) {
        cEne->cd();
        gStyle->SetOptStat("emr");
        obj1pX->GetXaxis()->SetNdivisions(40306, kFALSE);
        obj1pY->GetXaxis()->SetNdivisions(6, kFALSE);

        imgName = htmlDir + imgNameBXproj[iCanvas-1];
        obj1pX->SetStats(kTRUE);
        obj1pX->Draw("pe");
        cEne->Update();
        cEne->SaveAs(imgName.c_str());

        imgName = htmlDir + imgNameBYproj[iCanvas-1];
        obj1pY->SetStats(kTRUE);
        obj1pY->Draw("pe");
        cEne->Update();
        cEne->SaveAs(imgName.c_str());
      }
    }
  }

  imgNameBMap[3] = "";

  obj2f = h03_;

  if ( obj2f ) {

    meName = obj2f->GetName();

    replace(meName.begin(), meName.end(), ' ', '_');
    imgNameBMap[3] = meName + ".png";
    imgName = htmlDir + imgNameBMap[3];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions( 40118, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(170102, kFALSE);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->Draw("colz");
    labelGrid.Draw("text,same");
    Xaxis.Draw();
    cMap->Update();
    obj2f->GetXaxis()->SetLabelColor(0);
    cMap->SaveAs(imgName.c_str());
    obj2f->GetXaxis()->SetLabelColor(1);

    char projXName[100];
    char projYName[100];
    sprintf(projXName,"%s_px",meName.c_str());
    imgNameBXproj[3] = string(projXName) + ".png";
    sprintf(projYName,"%s_py",meName.c_str());
    imgNameBYproj[3] = string(projYName) + ".png";

    obj1fX = h03ProjEta_;
    obj1fY = h03ProjPhi_;

    if(obj1fX && obj1fY) {
      cEne->cd();
      gStyle->SetOptStat("emr");
      obj1fX->GetXaxis()->SetNdivisions(40306, kFALSE);
      obj1fY->GetXaxis()->SetNdivisions(6, kFALSE);

      imgName = htmlDir + imgNameBXproj[3];
      obj1fX->SetStats(kTRUE);
      obj1fX->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());

      imgName = htmlDir + imgNameBYproj[3];
      obj1fY->SetStats(kTRUE);
      obj1fY->Draw("pe");
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
    }
  }

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    if ( imgNameBMap[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameBMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 3; iCanvas <= 4; iCanvas++ ) {

    if ( imgNameBMap[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameBMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // projections X...
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 4; iCanvas++ ) {

    if ( imgNameBXproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameBXproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // projections Y...
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 4; iCanvas++ ) {

    if ( imgNameBYproj[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameBYproj[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  //   // super clusters
  //
  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameS[iCanvas-1] = "";

    obj1f = i01_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameS[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameS[iCanvas-1];

      cEne->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
    }
  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"sc_plots\"> <B> Super Clusters plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;


  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    if ( imgNameS[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameS[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // ===========================================================================
  // Higher Level variables
  // ===========================================================================

  for( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameHL[iCanvas-1] = "";

    obj1f = s01_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameHL[iCanvas-1] = meName + ".png";
      imgName = htmlDir + imgNameHL[iCanvas-1];

      cEne->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);

      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }

      obj1f->Draw();
      cEne->Update();
      cEne->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
    }
  }

  ///*****************************************************************************///
  htmlFile << "<br>" << endl;
  htmlFile <<  "<a name=\"hl_plots\"> <B> Higher Level Quantities plots </B> </a> " << endl;
  htmlFile << "</br>" << endl;
  ///*****************************************************************************///

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  // cluster shapes and invariant mass
  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    if ( imgNameHL[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameHL[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cEne;
  delete cMap;

  gStyle->SetPaintTextFormat();

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

