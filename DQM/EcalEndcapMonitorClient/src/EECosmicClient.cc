/*
 * \file EECosmicClient.cc
 *
 * $Date: 2008/03/15 14:07:45 $
 * $Revision: 1.54 $
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
#include "TGraph.h"
#include "TLine.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EECosmicClient::EECosmicClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
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

EECosmicClient::~EECosmicClient(){

}

void EECosmicClient::beginJob(DQMStore* dbe){

  dbe_ = dbe;

  if ( verbose_ ) cout << "EECosmicClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EECosmicClient::beginRun(void){

  if ( verbose_ ) cout << "EECosmicClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EECosmicClient::endJob(void) {

  if ( verbose_ ) cout << "EECosmicClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EECosmicClient::endRun(void) {

  if ( verbose_ ) cout << "EECosmicClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EECosmicClient::setup(void) {

}

void EECosmicClient::cleanup(void) {

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

bool EECosmicClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

        if ( ! Numbers::validEE(ism, jx, jy) ) continue;

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ix, iy));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(ix, iy);
            rms01  = h01_[ism-1]->GetBinError(ix, iy);
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ix, iy));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(ix, iy);
            rms02  = h02_[ism-1]->GetBinError(ix, iy);
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( Numbers::icEE(ism, jx, jy) == 1 ) {

            cout << "Preparing dataset for " << Numbers::sEE(ism) << " (ism=" << ism << ")" << endl;

            cout << "Cut (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "Sel (" << Numbers::ix0EE(i+1)+ix << "," << Numbers::iy0EE(i+1)+iy << ") " << num02  << " " << mean02 << " " << rms02  << endl;

            cout << endl;

          }

          o.setEventsOverLowThreshold(int(num01));
          o.setEventsOverHighThreshold(int(num02));

          o.setAvgEnergy(mean01);

          int ic = Numbers::indexEE(ism, jx, jy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EE_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
            dataset[ecid] = o;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonOccupancyDat ..." << endl;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EECosmicClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EECosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "EcalEndcap/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    sprintf(histo, "EcalEndcap/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    sprintf(histo, "EcalEndcap/EECosmicTask/Spectrum/EECT 1x1 energy spectrum %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03_[ism-1] );
    meh03_[ism-1] = me;

    sprintf(histo, "EcalEndcap/EECosmicTask/Spectrum/EECT 3x3 energy spectrum %s", Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    h04_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h04_[ism-1] );
    meh04_[ism-1] = me;

  }

}

void EECosmicClient::htmlOutput(int run, string& htmlDir, string& htmlName){

  cout << "Preparing EECosmicClient html output ..." << endl;

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
             << Numbers::sEE(superModules_[i]) << ">"
             << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2S labelGrid("labelGrid","label grid", 100, -2., 98., 100, -2., 98.);
  for ( short j=0; j<400; j++ ) {
    int x = 5*(1 + j%20);
    int y = 5*(1 + j/20);
    labelGrid.SetBinContent(x, y, Numbers::inTowersEE[j]);
  }
  labelGrid.SetMarkerSize(1);
  labelGrid.SetMinimum(0.1);

  string imgNameME[4], imgName, meName;

  TCanvas* cMe = new TCanvas("cMe", "Temp", 2*csize, 2*csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);

  TProfile2D* objp;
  TH1F* obj1f;

  // Loop on endcap sectors

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
        cMe->SetGridx();
        cMe->SetGridy();
        objp->GetXaxis()->SetLabelSize(0.02);
        objp->GetXaxis()->SetTitleSize(0.02);
        objp->GetYaxis()->SetLabelSize(0.02);
        objp->GetYaxis()->SetTitleSize(0.02);
        objp->GetZaxis()->SetLabelSize(0.02);
        objp->GetZaxis()->SetTitleSize(0.02);
        objp->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindFixBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindFixBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        cMe->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
        for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
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
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cAmp->Update();
      cAmp->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

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
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cAmp->Update();
      cAmp->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
             << Numbers::sEE(ism) << """></a><strong>"
             << Numbers::sEE(ism) << "</strong></h3>" << endl;
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

