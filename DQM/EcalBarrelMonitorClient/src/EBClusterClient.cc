/*
 * \file EBClusterClient.cc
 *
 * $Date: 2007/02/01 15:25:24 $
 * $Revision: 1.9 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBClusterClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBClusterClient::EBClusterClient(const ParameterSet& ps){

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

}

EBClusterClient::~EBClusterClient(){

}

void EBClusterClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EBClusterClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {


  }

}

void EBClusterClient::beginRun(void){

  if ( verbose_ ) cout << "EBClusterClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBClusterClient::endJob(void) {

  if ( verbose_ ) cout << "EBClusterClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBClusterClient::endRun(void) {

  if ( verbose_ ) cout << "EBClusterClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBClusterClient::setup(void) {

  mui_->setCurrentFolder( "EcalBarrel/EBClusterClient" );

}

void EBClusterClient::cleanup(void) {

  mui_->setCurrentFolder( "EcalBarrel/EBClusterClient" );

}

bool EBClusterClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, int ism) {

  bool status = true;

  return status;

}

void EBClusterClient::subscribe(void){

  if ( verbose_ ) cout << "EBClusterClient: subscribe" << endl;

  Char_t histo[200];

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster crystals");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy map");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number map");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster size");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy map");
  mui_->subscribe(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number map");
  mui_->subscribe(histo);

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBClusterClient: collate" << endl;

    sprintf(histo, "EBCLT basic cluster energy");
    me_h01_[0] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy");
    mui_->add(me_h01_[0], histo);

    sprintf(histo, "EBCLT basic cluster number");
    me_h01_[1] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number");
    mui_->add(me_h01_[1], histo);

    sprintf(histo, "EBCLT basic cluster crystals");
    me_h01_[2] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster crystals");
    mui_->add(me_h01_[2], histo);

    sprintf(histo, "EBCLT basic cluster energy map");
    me_h02_ = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy map");
    mui_->add(me_h02_, histo);

    sprintf(histo, "EBCLT basic cluster number map");
    me_h03_ = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number map");
    mui_->add(me_h03_, histo);

    sprintf(histo, "EBCLT super cluster energy");
    me_i01_[0] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy");
    mui_->add(me_i01_[0], histo);

    sprintf(histo, "EBCLT super cluster number");
    me_i01_[1] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number");
    mui_->add(me_i01_[1], histo);

    sprintf(histo, "EBCLT super cluster size");
    me_i01_[2] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster size");
    mui_->add(me_i01_[2], histo);

    sprintf(histo, "EBCLT super cluster energy map");
    me_i02_ = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy map");
    mui_->add(me_i02_, histo);

    sprintf(histo, "EBCLT super cluster number map");
    me_i03_ = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBClusterTask");
    sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number map");
    mui_->add(me_i03_, histo);

  }

}

void EBClusterClient::subscribeNew(void){

  Char_t histo[200];

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster crystals");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy map");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number map");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster size");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy map");
  mui_->subscribeNew(histo);

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number map");
  mui_->subscribeNew(histo);

}

void EBClusterClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBClusterClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBClusterClient: uncollate" << endl;

    if ( mui_ ) {

      mui_->removeCollate(me_h01_[0]);
      mui_->removeCollate(me_h01_[1]);
      mui_->removeCollate(me_h01_[2]);
      mui_->removeCollate(me_h02_);
      mui_->removeCollate(me_h03_);

      mui_->removeCollate(me_i01_[0]);
      mui_->removeCollate(me_i01_[1]);
      mui_->removeCollate(me_i01_[2]);
      mui_->removeCollate(me_i02_);
      mui_->removeCollate(me_i03_);

    }

  }

  Char_t histo[200];

  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster crystals");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster energy map");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT basic cluster number map");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster size");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster energy map");
  mui_->unsubscribe(histo);
  
  sprintf(histo, "*/EcalBarrel/EBClusterTask/EBCLT super cluster number map");
  mui_->unsubscribe(histo);

}

void EBClusterClient::softReset(void){

}

void EBClusterClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBClusterClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[200];

  MonitorElement* me;

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT basic cluster energy");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT basic cluster energy").c_str());
  }
  me = mui_->get(histo);
  h01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT basic cluster number");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT basic cluster number").c_str());
  }
  me = mui_->get(histo);
  h01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT basic cluster crystals");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT basic cluster crystals").c_str());
  }
  me = mui_->get(histo);
  h01_[2] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, h01_[2] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT basic cluster energy map");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT basic cluster energy map").c_str());
  }
  me = mui_->get(histo);
  h02_ = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_ );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT basic cluster number map");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT basic cluster number map").c_str());
  }
  me = mui_->get(histo);
  h03_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, h03_ );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT super cluster energy");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT super cluster energy").c_str());
  }
  me = mui_->get(histo);
  i01_[0] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[0] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT super cluster number");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT super cluster number").c_str());
  }
  me = mui_->get(histo);
  i01_[1] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[1] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT super cluster size");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT super cluster size").c_str());
  }
  me = mui_->get(histo);
  i01_[2] = EBMUtilsClient::getHisto<TH1F*>( me, cloneME_, i01_[2] );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT super cluster energy map");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT super cluster energy map").c_str());
  }
  me = mui_->get(histo);
  i02_ = EBMUtilsClient::getHisto<TProfile2D*>( me, cloneME_, i02_ );

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EBClusterTask/EBCLT super cluster number map");
  } else {
    sprintf(histo, (prefixME_+"EcalBarrel/EBClusterTask/EBCLT super cluster number map").c_str());
  }
  me = mui_->get(histo);
  i03_ = EBMUtilsClient::getHisto<TH2F*>( me, cloneME_, i03_ );

}

void EBClusterClient::htmlOutput(int run, string htmlDir, string htmlName){

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
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  string imgNameB[3], imgNameBMap[2], imgNameS[3], imgNameSMap[2], imgName, meName;

  TCanvas* cEne = new TCanvas("cEne", "Temp", csize, csize);
  TCanvas* cMap = new TCanvas("cMap", "Temp", csize, 2*csize);

  TH1F* obj1f = 0;
  TProfile2D* objp;
  TH2F* obj2f = 0;

  // basic clusters

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameB[iCanvas-1] = "";

    obj1f = h01_[iCanvas-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
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

  imgNameBMap[0] = "";

  objp = h02_;

  if ( objp ) {

    meName = objp->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" ); 
      }
    }
    imgNameBMap[0] = meName + ".png";
    imgName = htmlDir + imgNameBMap[0];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    objp->GetXaxis()->SetNdivisions(2);
    objp->GetYaxis()->SetNdivisions(18, kFALSE);
    cMap->SetGridx();
    cMap->SetGridy();
    cMap->SetRightMargin(0.15);
    if ( objp->GetMaximum(histMax) > 0. ) {
      gPad->SetLogz(1);
    } else {
      gPad->SetLogz(0);
    }
    objp->Draw("colz");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());
    gPad->SetLogz(0);

  }

  imgNameBMap[1] = "";

  obj2f = h03_;

  if ( obj2f ) {
  
    meName = obj2f->GetName();
  
    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameBMap[1] = meName + ".png";
    imgName = htmlDir + imgNameBMap[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions(2);
    obj2f->GetYaxis()->SetNdivisions(18, kFALSE);
    cMap->SetGridx();
    cMap->SetGridy();
    cMap->SetRightMargin(0.15);
    if ( obj2f->GetMaximum(histMax) > 0. ) {
      gPad->SetLogz(1); 
    } else {
      gPad->SetLogz(0);
    }
    obj2f->Draw("colz");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());
    gPad->SetLogz(0);
    
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

  // super clusters

  for ( int iCanvas = 1; iCanvas <= 3; iCanvas++ ) {

    imgNameS[iCanvas-1] = "";

    obj1f = i01_[iCanvas-1];

    if ( obj1f ) {
    
      meName = obj1f->GetName();
    
      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
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

  imgNameSMap[0] = "";
  
  objp = i02_;
  
  if ( objp ) {
  
    meName = objp->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" ); 
      }
    }
    imgNameSMap[0] = meName + ".png";
    imgName = htmlDir + imgNameSMap[0];
    
    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    objp->GetXaxis()->SetNdivisions(2);
    objp->GetYaxis()->SetNdivisions(18, kFALSE);
    cMap->SetGridx();
    cMap->SetGridy();
    cMap->SetRightMargin(0.15);
    if ( objp->GetMaximum(histMax) > 0. ) {
      gPad->SetLogz(1);
    } else {
      gPad->SetLogz(0);
    }
    objp->Draw("colz");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());
    gPad->SetLogz(0);

  }

  imgNameSMap[1] = "";

  obj2f = i03_;

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" ); 
      }
    }
    imgNameSMap[1] = meName + ".png";
    imgName = htmlDir + imgNameSMap[1];

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(10, pCol4);
    obj2f->GetXaxis()->SetNdivisions(2);
    obj2f->GetYaxis()->SetNdivisions(18, kFALSE);
    cMap->SetGridx();
    cMap->SetGridy();
    cMap->SetRightMargin(0.15);
    if ( obj2f->GetMaximum(histMax) > 0. ) {
      gPad->SetLogz(1);
    } else { 
      gPad->SetLogz(0);
    }
    obj2f->Draw("colz");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());
    gPad->SetLogz(0);

  }

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

    if ( imgNameSMap[iCanvas-1].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameSMap[iCanvas-1] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cEne;
  delete cMap;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

