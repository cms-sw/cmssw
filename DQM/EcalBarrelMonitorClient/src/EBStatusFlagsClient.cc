/*
 * \file EBStatusFlagsClient.cc
 *
 * $Date: 2008/04/07 07:24:32 $
 * $Revision: 1.14 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TCanvas.h"
#include "TStyle.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBStatusFlagsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBStatusFlagsClient::EBStatusFlagsClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

    h02_[ism-1] = 0;

    meh02_[ism-1] = 0;

  }

}

EBStatusFlagsClient::~EBStatusFlagsClient(){

}

void EBStatusFlagsClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( debug_ ) cout << "EBStatusFlagsClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBStatusFlagsClient::beginRun(void){

  if ( debug_ ) cout << "EBStatusFlagsClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBStatusFlagsClient::endJob(void) {

  if ( debug_ ) cout << "EBStatusFlagsClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBStatusFlagsClient::endRun(void) {

  if ( debug_ ) cout << "EBStatusFlagsClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBStatusFlagsClient::setup(void) {

  dbe_->setCurrentFolder( "EcalBarrel/EBStatusFlagsClient" );

}

void EBStatusFlagsClient::cleanup(void) {

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

  dbe_->setCurrentFolder( "EcalBarrel/EBStatusFlagsClient" );

}

bool EBStatusFlagsClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
    cout << endl;

    UtilsClient::printBadChannels(meh01_[ism-1], UtilsClient::getHisto<TH2F*>(meh01_[ism-1]), true);

    if ( meh01_[ism-1]->getEntries() != 0 ) status = false;

  }

  return status;

}

void EBStatusFlagsClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBStatusFlagsClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "EcalBarrel/EBStatusFlagsTask/FEStatus/EBSFT front-end status %s", Numbers::sEB(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, "EcalBarrel/EBStatusFlagsTask/FEStatus/EBSFT front-end status bits %s", Numbers::sEB(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

  }

}

void EBStatusFlagsClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EBStatusFlagsClient html output ..." << endl;

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
             << Numbers::sEB(superModules_[i]) << ">"
             << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol5[10];
  for ( int i = 0; i < 10; i++ ) pCol5[i] = 501+i;

  TH2C dummy( "dummy", "dummy for sm", 17, 0., 17., 4, 0., 4. );
  for ( int i = 0; i < 68; i++ ) {
    int a = ( i/4 );
    int b = ( i%4 );
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(0.1);

  string imgNameQual, imgNameBits, imgName, meName;

  TCanvas* cStatus = new TCanvas("cStatus", "Temp", 3*csize, csize);
  TCanvas* cStatusBits = new TCanvas("cStatusBits", "Temp", 3*csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;

  // Loop on barrel supermodules

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
      obj2f->GetXaxis()->SetNdivisions(17);
      obj2f->GetYaxis()->SetNdivisions(4);
      cStatus->SetGridx();
      cStatus->SetGridy();
      obj2f->SetMinimum(0.0);
      obj2f->Draw("colz");
      dummy.Draw("text,same");
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
             << Numbers::sEB(ism) << """></a><strong>"
             << Numbers::sEB(ism) << "</strong></h3>" << endl;
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

