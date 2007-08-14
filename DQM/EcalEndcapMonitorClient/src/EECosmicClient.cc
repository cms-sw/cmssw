/*
 * \file EECosmicClient.cc
 *
 * $Date: 2007/08/09 14:36:55 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EECosmicClient::EECosmicClient(const ParameterSet& ps){

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
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;

  }

}

EECosmicClient::~EECosmicClient(){

}

void EECosmicClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EECosmicClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EECosmicClient::beginRun(void){

  if ( verbose_ ) cout << "EECosmicClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EECosmicClient::endJob(void) {

  if ( verbose_ ) cout << "EECosmicClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EECosmicClient::endRun(void) {

  if ( verbose_ ) cout << "EECosmicClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EECosmicClient::setup(void) {

}

void EECosmicClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    meh01_[ism-1] = 0;
    meh02_[ism-1] = 0;
    meh03_[ism-1] = 0;

  }

}

bool EECosmicClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  EcalLogicID ecid;

  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

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

          if ( ix == 1 && iy == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "Sel (" << ix << "," << iy << ") " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "Cut (" << ix << "," << iy << ") " << num02  << " " << mean02 << " " << rms02  << endl;

            cout << endl;

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          int ic = Numbers::icEE(ism, ix, iy);

          if ( ic == -1 ) continue;

          if ( econn ) {
            try {
              ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalEndcap), ic);
              dataset[ecid] = o;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting MonOccupancyDat ... " << flush;
      if ( dataset.size() != 0 ) econn->insertDataArraySet(&dataset, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return status;

}

void EECosmicClient::subscribe(void){

  if ( verbose_ ) cout << "EECosmicClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Spectrum/EECT energy spectrum %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EECosmicClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EECT energy sel %s", Numbers::sEE(ism).c_str());
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EECosmicTask/Sel");
      sprintf(histo, "*/EcalEndcap/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EECT energy cut %s", Numbers::sEE(ism).c_str());
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EECosmicTask/Cut");
      sprintf(histo, "*/EcalEndcap/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EECT energy spectrum %s", Numbers::sEE(ism).c_str());
      me_h03_[ism-1] = mui_->collate1D(histo, histo, "EcalEndcap/Sums/EECosmicTask/Spectrum");
      sprintf(histo, "*/EcalEndcap/EECosmicTask/Spectrum/EECT energy spectrum %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h03_[ism-1], histo);

    }

  }

}

void EECosmicClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Spectrum/EECT energy spectrum %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo);

  }

}

void EECosmicClient::unsubscribe(void){

  if ( verbose_ ) cout << "EECosmicClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EECosmicClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo);
    sprintf(histo, "*/EcalEndcap/EECosmicTask/Spectrum/EECT energy spectrum %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo);

  }

}

void EECosmicClient::softReset(void){

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meh01_[ism-1] ) mui_->softReset(meh01_[ism-1]);
    if ( meh02_[ism-1] ) mui_->softReset(meh02_[ism-1]);
    if ( meh03_[ism-1] ) mui_->softReset(meh03_[ism-1]);

  }

}

void EECosmicClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EECosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EECosmicTask/Sel/EECT energy sel %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EECosmicTask/Sel/EECT energy sel %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EECosmicTask/Cut/EECT energy cut %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EECosmicTask/Cut/EECT energy cut %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );
    meh02_[ism-1] = me;

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EECosmicTask/Spectrum/EECT energy spectrum %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EECosmicTask/Spectrum/EECT energy spectrum %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h03_[ism-1] = UtilsClient::getHisto<TH1F*>( me, cloneME_, h03_[ism-1] );
    meh03_[ism-1] = me;

  }

}

void EECosmicClient::htmlOutput(int run, string htmlDir, string htmlName){

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
	     << Numbers::sEE(superModules_[i]).c_str() << ">"
	     << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  string imgNameME[3], imgName, meName;

  TCanvas* cMe = new TCanvas("cMe", "Temp", 2*csize, 2*csize);
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

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        cMe->SetGridx();
        cMe->SetGridy();
        objp->GetXaxis()->SetLabelSize(0.02);
        objp->GetYaxis()->SetLabelSize(0.02);
        objp->Draw("colz");
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

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
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

    if( i>0 ) htmlFile << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile << "<hr>" << std::endl;
    htmlFile << "<h3><a name="""
	     << Numbers::sEE(ism).c_str() << """></a><strong>"
	     << Numbers::sEE(ism).c_str() << "</strong></h3>" << endl;
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

    if ( imgNameME[2].size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[2] << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

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

