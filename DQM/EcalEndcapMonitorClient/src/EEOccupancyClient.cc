/*
 * \file EEOccupancyClient.cc
 *
 * $Date: 2008/04/07 09:00:42 $
 * $Revision: 1.21 $
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

#include "DQMServices/Core/interface/DQMStore.h"

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

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

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

void EEOccupancyClient::beginJob(DQMStore* dqmStore){

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EEOccupancyClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEOccupancyClient::beginRun(void){

  if ( debug_ ) cout << "EEOccupancyClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEOccupancyClient::endJob(void) {

  if ( debug_ ) cout << "EEOccupancyClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::endRun(void) {

  if ( debug_ ) cout << "EEOccupancyClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEOccupancyClient::setup(void) {

  dqmStore_->setCurrentFolder( "EcalEndcap/EEOccupancyClient" );

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
    if ( debug_ ) cout << "EEOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE -");
  me = dqmStore_->get(histo);
  h01_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE - projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE - projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE +");
  me = dqmStore_->get(histo);
  h01_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE + projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE + projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE -");
  me = dqmStore_->get(histo);
  h01_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE +");
  me = dqmStore_->get(histo);
  h01_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE -");
  me = dqmStore_->get(histo);
  h01_[0][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE - projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[0][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE - projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[0][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE +");
  me = dqmStore_->get(histo);
  h01_[1][2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE + projection R");
  me = dqmStore_->get(histo);
  h01ProjR_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjR_[1][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi occupancy EE + projection phi");
  me = dqmStore_->get(histo);
  h01ProjPhi_[1][2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1][2] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE -");
  me = dqmStore_->get(histo);
  h02_[0][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection R");
  me = dqmStore_->get(histo);
  h02ProjR_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi");
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE +");
  me = dqmStore_->get(histo);
  h02_[1][0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection R");
  me = dqmStore_->get(histo);
  h02ProjR_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi");
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][0] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE -");
  me = dqmStore_->get(histo);
  h02_[0][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection R");
  me = dqmStore_->get(histo);
  h02ProjR_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection phi");
  me = dqmStore_->get(histo);
  h02ProjPhi_[0][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE +");
  me = dqmStore_->get(histo);
  h02_[1][1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection R");
  me = dqmStore_->get(histo);
  h02ProjR_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjR_[1][1] );

  sprintf(histo, "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection phi");
  me = dqmStore_->get(histo);
  h02ProjPhi_[1][1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1][1] );

}

void EEOccupancyClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EEOccupancyClient html output ..." << endl;

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

  // Produce the plots to be shown as .png files from existing histograms

  const int csize1D = 250;
  const int csize2D = 500;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy1("labelGrid1","label grid for EE -", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      dummy1.SetBinContent(i, j, -10);
    }
  }

  dummy1.SetBinContent(2, 5, -3);
  dummy1.SetBinContent(2, 7, -2);
  dummy1.SetBinContent(4, 9, -1);
  dummy1.SetBinContent(7, 9, -9);
  dummy1.SetBinContent(9, 7, -8);
  dummy1.SetBinContent(9, 5, -7);
  dummy1.SetBinContent(8, 3, -6);
  dummy1.SetBinContent(5, 2, -5);
  dummy1.SetBinContent(3, 3, -4);

  dummy1.SetMarkerSize(2);
  dummy1.SetMinimum(-9.01);
  dummy1.SetMaximum(-0.01);

  TH2C dummy2("labelGrid2","label grid for EE +", 10, 0., 100., 10, 0., 100.);

  for ( int i=1; i<=10; i++) {
    for ( int j=1; j<=10; j++) {
      dummy2.SetBinContent(i, j, -10);
    }
  }

  dummy2.SetBinContent(2, 5, +7);
  dummy2.SetBinContent(2, 7, +8);
  dummy2.SetBinContent(4, 9, +9);
  dummy2.SetBinContent(7, 9, +1);
  dummy2.SetBinContent(9, 7, +2);
  dummy2.SetBinContent(9, 5, +3);
  dummy2.SetBinContent(8, 3, +4);
  dummy2.SetBinContent(6, 2, +5);
  dummy2.SetBinContent(3, 3, +6);

  dummy2.SetMarkerSize(2);
  dummy2.SetMinimum(+0.01);
  dummy2.SetMaximum(+9.01);

  string imgNameMap[2][3], imgNameProjR[2][3], imgNameProjPhi[2][3];
  string imgNameMapThr[2][2], imgNameProjRThr[2][2], imgNameProjPhiThr[2][2];
  string imgName, meName;

  TCanvas* cMap = new TCanvas("cMap", "cMap", csize2D, csize2D);
  TCanvas* cProj = new TCanvas("cProj", "cProj", csize1D, csize1D);

  TH2F* obj2f;
  TH1F* obj1fR;
  TH1F* obj1fPhi;

  gStyle->SetPaintTextFormat("+g");

  // Occupancy without threshold
  for ( int iMap=0; iMap<3; iMap++ ) {
    for ( int iEE=0; iEE<2; iEE++ ) {

      imgNameMap[iEE][iMap] = "";

      obj2f = h01_[iEE][iMap];

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMap[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameMap[iEE][iMap];

        cMap->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(10, kFALSE);
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetNdivisions(10, kFALSE);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        cMap->SetGridx();
        cMap->SetGridy();
        obj2f->Draw("colz");
        if ( iEE == 0 ) dummy1.Draw("text,same");
        if ( iEE == 1 ) dummy2.Draw("text,same");
        cMap->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        cMap->Update();
        cMap->SaveAs(imgName.c_str());

      }

      obj1fR = h01ProjR_[iEE][iMap];

      if ( obj1fR ) {

        meName = obj1fR->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameProjR[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameProjR[iEE][iMap];

        cProj->cd();
        gStyle->SetOptStat("emr");
        obj1fR->SetStats(kTRUE);
        obj1fR->Draw("pe");
        cProj->Update();
        cProj->SaveAs(imgName.c_str());

      }

      obj1fPhi = h01ProjPhi_[iEE][iMap];

      if ( obj1fPhi ) {

        meName = obj1fPhi->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameProjPhi[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameProjPhi[iEE][iMap];

        cProj->cd();
        gStyle->SetOptStat("emr");
        obj1fPhi->GetXaxis()->SetNdivisions(50206, kFALSE);
        obj1fPhi->SetStats(kTRUE);
        obj1fPhi->Draw("pe");
        cProj->Update();
        cProj->SaveAs(imgName.c_str());

      }

    }
  }

  // Occupancy with threshold
  for ( int iMap=0; iMap<2; iMap++ ) {
    for( int iEE=0; iEE<2; iEE++ ) {

      imgNameMapThr[iEE][iMap] = "";

      obj2f = h02_[iEE][iMap];

      if ( obj2f ) {

        meName = obj2f->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameMapThr[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameMapThr[iEE][iMap];

        cMap->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(10, kFALSE);
        obj2f->GetXaxis()->SetLabelSize(0.02);
        obj2f->GetXaxis()->SetTitleSize(0.02);
        obj2f->GetYaxis()->SetNdivisions(10, kFALSE);
        obj2f->GetYaxis()->SetLabelSize(0.02);
        obj2f->GetYaxis()->SetTitleSize(0.02);
        obj2f->GetZaxis()->SetLabelSize(0.02);
        cMap->SetGridx();
        cMap->SetGridy();
        obj2f->Draw("colz");
        if ( iEE == 0 ) dummy1.Draw("text,same");
        if ( iEE == 1 ) dummy2.Draw("text,same");
        cMap->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        cMap->Update();
        cMap->SaveAs(imgName.c_str());

      }

      obj1fR = h02ProjR_[iEE][iMap];

      if ( obj1fR ) {

        meName = obj1fR->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameProjRThr[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameProjRThr[iEE][iMap];

        cProj->cd();
        gStyle->SetOptStat("emr");
        obj1fR->SetStats(kTRUE);
        obj1fR->Draw("pe");
        cProj->Update();
        cProj->SaveAs(imgName.c_str());

      }

      obj1fPhi = h02ProjPhi_[iEE][iMap];

      if ( obj1fPhi ) {

        meName = obj1fPhi->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameProjPhiThr[iEE][iMap] = meName + ".png";
        imgName = htmlDir + imgNameProjPhiThr[iEE][iMap];

        cProj->cd();
        gStyle->SetOptStat("emr");
        obj1fPhi->GetXaxis()->SetNdivisions(50206, kFALSE);
        obj1fPhi->SetStats(kTRUE);
        obj1fPhi->Draw("pe");
        cProj->Update();
        cProj->SaveAs(imgName.c_str());

      }

    }
  }

  gStyle->SetPaintTextFormat();

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<3; iMap++) {
    for (int iEE=0; iEE<2; iEE++) {
      if ( imgNameMap[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMap[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<3; iMap++) {
    for (int iEE=0; iEE<2; iEE++) {
      if ( imgNameProjR[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameProjR[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
      if ( imgNameProjPhi[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameProjPhi[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<2; iMap++) {
    for (int iEE=0; iEE<2; iEE++) {
      if ( imgNameMapThr[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMapThr[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<2; iMap++) {
    for (int iEE=0; iEE<2; iEE++) {
      if ( imgNameProjRThr[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameProjRThr[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
      if ( imgNameProjPhiThr[iEE][iMap].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameProjPhiThr[iEE][iMap] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile.close();

}

