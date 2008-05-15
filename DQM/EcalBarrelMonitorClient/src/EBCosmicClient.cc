/*
 * \file EBCosmicClient.cc
 *
 * $Date: 2008/04/08 15:06:21 $
 * $Revision: 1.112 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TCanvas.h"
#include "TStyle.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBCosmicClient::EBCosmicClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;
    meh04_[ism-1] = 0;

  }

}

EBCosmicClient::~EBCosmicClient(){

}

void EBCosmicClient::beginJob(DQMStore* dqmStore){

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBCosmicClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBCosmicClient::beginRun(void){

  if ( debug_ ) cout << "EBCosmicClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBCosmicClient::endJob(void) {

  if ( debug_ ) cout << "EBCosmicClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBCosmicClient::endRun(void) {

  if ( debug_ ) cout << "EBCosmicClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBCosmicClient::setup(void) {

}

void EBCosmicClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;
    meh04_[ism-1] = 0;

  }

}

bool EBCosmicClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(ie, ip);
            rms01  = h01_[ism-1]->GetBinError(ie, ip);
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(ie, ip);
            rms02  = h02_[ism-1]->GetBinError(ie, ip);
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "Cut (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
              cout << "Sel (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;
              cout << endl;
            }

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset[ecid] = o;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonOccupancyDat ..." << endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EBCosmicClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBCosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Cut/EBCT energy cut %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Sel/EBCT energy sel %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Spectrum/EBCT 1x1 energy spectrum %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03_[ism-1] );
    meh03_[ism-1] = me;

    sprintf(histo, (prefixME_ + "/EBCosmicTask/Spectrum/EBCT 3x3 energy spectrum %s").c_str(), Numbers::sEB(ism).c_str());
    me = dqmStore_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h04_[ism-1] );
    meh04_[ism-1] = me;

  }

}

void EBCosmicClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  if ( verbose_ ) cout << "Preparing EBCosmicClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:CosmicTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">COSMIC</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;
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

  string imgNameME[4], imgName, meName;

  TCanvas* cMe = new TCanvas("cMe", "Temp", 3*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);

  TProfile2D* objp;
  TH1F* obj1f;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Monitoring elements plots

    for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = h01_[ism-1];
          break;
        case 2:
          objp = h02_[ism-1];
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        replace(meName.begin(), meName.end(), ' ', '_');
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        objp->GetXaxis()->SetNdivisions(17);
        objp->GetYaxis()->SetNdivisions(4);
        cMe->SetGridx();
        cMe->SetGridy();
        objp->Draw("colz");
        dummy.Draw("text,same");
        cMe->Update();
        cMe->SaveAs(imgName.c_str());

      }

    }

    // Energy spectrum distributions

    imgNameME[2] = "";

    obj1f = h03_[ism-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameME[2] = meName + ".png";
      imgName = htmlDir + imgNameME[2];

      cAmp->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(kTRUE);
      } else {
        gPad->SetLogy(kFALSE);
      }
      obj1f->Draw();
      cAmp->Update();
      cAmp->SaveAs(imgName.c_str());
      gPad->SetLogy(kFALSE);

    }

    imgNameME[3] = "";

    obj1f = h04_[ism-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameME[3] = meName + ".png";
      imgName = htmlDir + imgNameME[3];

      cAmp->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(kTRUE);
      } else {
        gPad->SetLogy(kFALSE);
      }
      obj1f->Draw();
      cAmp->Update();
      cAmp->SaveAs(imgName.c_str());
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

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 3 ; iCanvas <= 4 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cMe;
  delete cAmp;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

