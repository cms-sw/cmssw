/*
 * \file EBOccupancyClient.cc
 *
 * $Date: 2008/01/27 20:22:05 $
 * $Revision: 1.14 $
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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBOccupancyClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBOccupancyClient::EBOccupancyClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // enableMonitorDaemon_ switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", false);

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( int i=0; i<3; i++) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; i++) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

EBOccupancyClient::~EBOccupancyClient(){

}

void EBOccupancyClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EBOccupancyClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBOccupancyClient::beginRun(void){

  if ( verbose_ ) cout << "EBOccupancyClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBOccupancyClient::endJob(void) {

  if ( verbose_ ) cout << "EBOccupancyClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBOccupancyClient::endRun(void) {

  if ( verbose_ ) cout << "EBOccupancyClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBOccupancyClient::setup(void) {

  dbe_->setCurrentFolder( "EcalBarrel/EBOccupancyClient" );

}

void EBOccupancyClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {

    for ( int i=0; i<3; ++i ) {
      if ( h01_[i] ) delete h01_[i];
      if ( h01ProjEta_[i] ) delete h01ProjEta_[i];
      if ( h01ProjPhi_[i] ) delete h01ProjPhi_[i];
    }

    for ( int i=0; i<2; ++i ) {
      if ( h02_[i] ) delete h02_[i];
      if ( h02ProjEta_[i] ) delete h02ProjEta_[i];
      if ( h02ProjPhi_[i] ) delete h02ProjPhi_[i];
    }

  }

  for ( int i=0; i<3; ++i ) {
    h01_[i] = 0;
    h01ProjEta_[i] = 0;
    h01ProjPhi_[i] = 0;
  }

  for ( int i=0; i<2; ++i ) {
    h02_[i] = 0;
    h02ProjEta_[i] = 0;
    h02ProjPhi_[i] = 0;
  }

}

bool EBOccupancyClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EBOccupancyClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBOccupancyClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  char histo[200];

  MonitorElement* me;

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT digi occupancy").c_str());
  me = dbe_->get(histo);
  h01_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT digi occupancy projection eta").c_str());
  me = dbe_->get(histo);
  h01ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT digi occupancy projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy").c_str());
  me = dbe_->get(histo);
  h01_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[1] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection eta").c_str());
  me = dbe_->get(histo);
  h01ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[1] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[1] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi occupancy").c_str());
  me = dbe_->get(histo);
  h01_[2] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h01_[2] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi occupancy projection eta").c_str());
  me = dbe_->get(histo);
  h01ProjEta_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjEta_[2] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi occupancy projection phi").c_str());
  me = dbe_->get(histo);
  h01ProjPhi_[2] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h01ProjPhi_[2] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy").c_str());
  me = dbe_->get(histo);
  h02_[0] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection eta").c_str());
  me = dbe_->get(histo);
  h02ProjEta_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[0] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[0] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy").c_str());
  me = dbe_->get(histo);
  h02_[1] = UtilsClient::getHisto<TH2F*> ( me, cloneME_, h02_[1] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection eta").c_str());
  me = dbe_->get(histo);
  h02ProjEta_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjEta_[1] );

  sprintf(histo, (prefixME_+"EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection phi").c_str());
  me = dbe_->get(histo);
  h02ProjPhi_[1] = UtilsClient::getHisto<TH1F*> ( me, cloneME_, h02ProjPhi_[1] );

}

void EBOccupancyClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBOccupancyClient html output ..." << endl;

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
  const int csize2D = 300;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 401+i;

  TH2C dummy( "dummy", "dummy for eb", 18, 0., 360., 2, -85., 85.);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 1 + sm/18;
    dummy.SetBinContent(x, y, Numbers::iEB(sm+1));
  }
  dummy.SetMarkerSize(2);
  dummy.SetMinimum(-18.01);

  TH2C dummyTP( "dummyTP", "dummy for eb TP", 18, 0., 72., 2, -17., 17.);
  for ( short sm=0; sm<36; sm++ ) {
    int x = 1 + sm%18;
    int y = 1 + sm/18;
    dummyTP.SetBinContent(x, y, Numbers::iEB(sm+1));
  }
  dummyTP.SetMarkerSize(2);
  dummyTP.SetMinimum(-18.01);

  string imgNameMap[3], imgNameProjEta[3], imgNameProjPhi[3];
  string imgNameMapThr[2], imgNameProjEtaThr[2], imgNameProjPhiThr[2];
  string imgName, meName;

  TCanvas* cMap = new TCanvas("cMap", "cMap", int(360./170.*csize2D), csize2D);
  TCanvas* cProj = new TCanvas("cProj", "cProj", csize1D, csize1D);

  TH2F* obj2f;
  TH1F* obj1fEta;
  TH1F* obj1fPhi;

  gStyle->SetPaintTextFormat("+g");

  // Occupancy without threshold
  for ( int iMap=0; iMap<3; iMap++ ) {

    imgNameMap[iMap] = "";

    obj2f = h01_[iMap];

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameMap[iMap] = meName + ".png";
      imgName = htmlDir + imgNameMap[iMap];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
      obj2f->GetYaxis()->SetNdivisions(2);
      cMap->SetGridx();
      cMap->SetGridy();
      obj2f->Draw("colz");
      if ( iMap == 2 ) dummyTP.Draw("text,same");
      else dummy.Draw("text,same");
      cMap->Update();
      cMap->SaveAs(imgName.c_str());

    }

    obj1fEta = h01ProjEta_[iMap];

    if ( obj1fEta ) {

      meName = obj1fEta->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameProjEta[iMap] = meName + ".png";
      imgName = htmlDir + imgNameProjEta[iMap];

      cProj->cd();
      gStyle->SetOptStat("emr");
      obj1fEta->SetStats(kTRUE);
      obj1fEta->Draw("pe");
      cProj->Update();
      cProj->SaveAs(imgName.c_str());

    }

    obj1fPhi = h01ProjPhi_[iMap];

    if ( obj1fPhi ) {

      meName = obj1fPhi->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameProjPhi[iMap] = meName + ".png";
      imgName = htmlDir + imgNameProjPhi[iMap];

      cProj->cd();
      gStyle->SetOptStat("emr");
      obj1fPhi->SetStats(kTRUE);
      obj1fPhi->Draw("pe");
      cProj->Update();
      cProj->SaveAs(imgName.c_str());

    }

  }

  // Occupancy with threshold
  for ( int iMap=0; iMap<2; iMap++ ) {

    imgNameMapThr[iMap] = "";

    obj2f = h02_[iMap];

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameMapThr[iMap] = meName + ".png";
      imgName = htmlDir + imgNameMapThr[iMap];

      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      obj2f->GetXaxis()->SetNdivisions(18, kFALSE);
      obj2f->GetYaxis()->SetNdivisions(2);
      cMap->SetGridx();
      cMap->SetGridy();
      obj2f->Draw("colz");
      if ( iMap == 1 ) dummyTP.Draw("text,same");
      else dummy.Draw("text,same");
      cMap->Update();
      cMap->SaveAs(imgName.c_str());

    }

    obj1fEta = h02ProjEta_[iMap];

    if ( obj1fEta ) {

      meName = obj1fEta->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameProjEtaThr[iMap] = meName + ".png";
      imgName = htmlDir + imgNameProjEtaThr[iMap];

      cProj->cd();
      gStyle->SetOptStat("emr");
      obj1fEta->SetStats(kTRUE);
      obj1fEta->Draw("pe");
      cProj->Update();
      cProj->SaveAs(imgName.c_str());

    }

    obj1fPhi = h02ProjPhi_[iMap];

    if ( obj1fPhi ) {

      meName = obj1fPhi->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " ) {
          meName.replace(i, 1 ,"_" );
        }
      }

      imgNameProjPhiThr[iMap] = meName + ".png";
      imgName = htmlDir + imgNameProjPhiThr[iMap];

      cProj->cd();
      gStyle->SetOptStat("emr");
      obj1fPhi->SetStats(kTRUE);
      obj1fPhi->Draw("pe");
      cProj->Update();
      cProj->SaveAs(imgName.c_str());

    }

  }

  gStyle->SetPaintTextFormat();

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<3; iMap++) {
    if ( imgNameMap[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameMap[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<3; iMap++) {
    if ( imgNameProjEta[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameProjEta[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameProjPhi[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameProjPhi[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<2; iMap++) {
    if ( imgNameMapThr[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameMapThr[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  for (int iMap=0; iMap<2; iMap++) {
    if ( imgNameProjEtaThr[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameProjEtaThr[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    if ( imgNameProjPhiThr[iMap].size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameProjPhiThr[iMap] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile.close();

}

