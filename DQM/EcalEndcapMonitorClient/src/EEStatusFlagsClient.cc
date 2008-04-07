/*
 * \file EEStatusFlagsClient.cc
 *
 * $Date: 2008/04/07 08:44:21 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EEStatusFlagsClient::EEStatusFlagsClient(const ParameterSet& ps){

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

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

    h02_[ism-1] = 0;

    meh02_[ism-1] = 0;

  }

}

EEStatusFlagsClient::~EEStatusFlagsClient(){

}

void EEStatusFlagsClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( debug_ ) cout << "EEStatusFlagsClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EEStatusFlagsClient::beginRun(void){

  if ( debug_ ) cout << "EEStatusFlagsClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EEStatusFlagsClient::endJob(void) {

  if ( debug_ ) cout << "EEStatusFlagsClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EEStatusFlagsClient::endRun(void) {

  if ( debug_ ) cout << "EEStatusFlagsClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EEStatusFlagsClient::setup(void) {

  dbe_->setCurrentFolder( "EcalEndcap/EEStatusFlagsClient" );

}

void EEStatusFlagsClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;

  }

  dbe_->setCurrentFolder( "EcalEndcap/EEStatusFlagsClient" );

}

bool EEStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);
    }

    if ( meh01_[ism-1]->getEntries() != 0 ) status = false;

  }

  return status;

}

void EEStatusFlagsClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EEStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "EcalEndcap/EEStatusFlagsTask/FEStatus/EESFT front-end status %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, "EcalEndcap/EEStatusFlagsTask/FEStatus/EESFT front-end status bits %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

  }

}

void EEStatusFlagsClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EEStatusFlagsClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:StatusFlagsTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">DATA FLAGS</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;
  htmlFile << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile << "<td bgcolor=white><a href=""#"
             << Numbers::sEE(superModules_[i]) << ">"
             << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol5[10];
  for ( int i = 0; i < 10; i++ ) pCol5[i] = 501+i;

  TH2S labelGrid("labelGrid","label grid", 100, -2., 98., 100, -2., 98.);
  for ( short j=0; j<400; j++ ) {
    int x = 5*(1 + j%20);
    int y = 5*(1 + j/20);
    labelGrid.SetBinContent(x, y, Numbers::inTowersEE[j]);
  }
  labelGrid.SetMarkerSize(1);
  labelGrid.SetMinimum(0.1);

  string imgNameQual, imgNameBits, imgName, meName;

  TCanvas* cStatus = new TCanvas("cStatus", "Temp", 2*csize, 2*csize);
  TCanvas* cStatusBits = new TCanvas("cStatusBits", "Temp", 3*csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;

  // Loop on endcap sectors

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Quality plots

    imgNameQual = "";

    obj2f = h01_[ism-1];

    if ( obj2f ) {

      meName = obj2f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameQual = meName + ".png";
      imgName = htmlDir + imgNameQual;

      cStatus->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol5);
      cStatus->SetGridx();
      cStatus->SetGridy();
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
      int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
      int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
      int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
      labelGrid.GetXaxis()->SetRange(x1, x2);
      labelGrid.GetYaxis()->SetRange(y1, y2);
      labelGrid.Draw("text,same");
      cStatus->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cStatus->Update();
      cStatus->SaveAs(imgName.c_str());

    }

    imgNameBits = "";

    obj1f = h02_[ism-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameBits = meName + ".png";
      imgName = htmlDir + imgNameBits;

      cStatusBits->cd();
      gStyle->SetOptStat("e");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(kTRUE);
      } else {
        gPad->SetLogy(kFALSE);
      }
      gPad->SetBottomMargin(0.25);
      obj1f->GetXaxis()->LabelsOption("v");
      obj1f->GetXaxis()->SetLabelSize(0.05);
      obj1f->Draw();
      cStatusBits->Update();
      cStatusBits->SaveAs(imgName.c_str());
      gPad->SetLogy(kFALSE);

    }

    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
             << Numbers::sEE(ism) << """></a><strong>"
             << Numbers::sEE(ism) << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    if ( imgNameQual.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    if ( imgNameBits.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameBits << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cStatus;
  delete cStatusBits;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

