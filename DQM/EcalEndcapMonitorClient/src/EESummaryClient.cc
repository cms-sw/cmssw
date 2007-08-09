/*
 * \file EESummaryClient.cc
 *
 * $Date: 2007/06/24 09:37:59 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>

#include "TStyle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELaserClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETimingClient.h>

#include <DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EESummaryClient::EESummaryClient(const ParameterSet& ps){

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

  meIntegrity_      = 0;
  mePedestalOnline_ = 0;

  qtg01_ = 0;
  qtg02_ = 0;
  qtg03_ = 0;

}

EESummaryClient::~EESummaryClient(){

}

void EESummaryClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EESummaryClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    sprintf(qtname, "EEIT summary quality test");
    qtg01_ = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEOT summary quality test");
    qtg02_ = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    sprintf(qtname, "EEPOT summary quality test");
    qtg03_ = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

    qtg01_->setMeanRange(1., 6.);
    qtg02_->setMeanRange(1., 6.);
    qtg03_->setMeanRange(1., 6.);

    qtg01_->setErrorProb(1.00);
    qtg02_->setErrorProb(1.00);
    qtg03_->setErrorProb(1.00);

  }

}

void EESummaryClient::beginRun(void){

  if ( verbose_ ) cout << "EESummaryClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EESummaryClient::endJob(void) {

  if ( verbose_ ) cout << "EESummaryClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EESummaryClient::endRun(void) {

  if ( verbose_ ) cout << "EESummaryClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EESummaryClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EESummaryClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  if ( meIntegrity_ ) dbe->removeElement( meIntegrity_->getName() );
  sprintf(histo, "EEIT integrity quality summary");
  meIntegrity_ = dbe->book2D(histo, histo, 180, 0., 360., 170, -85., 85.);

  if ( mePedestalOnline_ ) dbe->removeElement( mePedestalOnline_->getName() );
  sprintf(histo, "EEPOT pedestal quality summary G12");
  mePedestalOnline_ = dbe->book2D(histo, histo, 180, 0., 360., 170, -85., 85.);

}

void EESummaryClient::cleanup(void) {

  mui_->setCurrentFolder( "EcalEndcap/EESummaryClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  if ( meIntegrity_ ) dbe->removeElement( meIntegrity_->getName() );
  meIntegrity_ = 0;

  if ( mePedestalOnline_ ) dbe->removeElement( mePedestalOnline_->getName() );
  mePedestalOnline_ = 0;

}

bool EESummaryClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

//  UtilsClient::printBadChannels(qtg01_);
//  UtilsClient::printBadChannels(qtg02_);
//  UtilsClient::printBadChannels(qtg03_);

  return status;

}

void EESummaryClient::subscribe(void){

  if ( verbose_ ) cout << "EESummaryClient: subscribe" << endl;

  Char_t histo[200];

  sprintf(histo, "EcalEndcap/EESummaryClient/EEIT integrity quality summary");
  if ( qtg01_ ) mui_->useQTest(histo, qtg01_->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEOT occupancy summary");
  if ( qtg02_ ) mui_->useQTest(histo, qtg02_->getName());
  sprintf(histo, "EcalEndcap/EESummaryClient/EEPOT pedestal quality summary G12");
  if ( qtg03_ ) mui_->useQTest(histo, qtg03_->getName());

}

void EESummaryClient::subscribeNew(void){

}

void EESummaryClient::unsubscribe(void){

  if ( verbose_ ) cout << "EESummaryClient: unsubscribe" << endl;

}

void EESummaryClient::softReset(void){

}

void EESummaryClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EESummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  for ( int iex = 1; iex <= 170; iex++ ) {
    for ( int ipx = 1; ipx <= 360; ipx++ ) {

      meIntegrity_->setBinContent( ipx, iex, -1. );
      mePedestalOnline_->setBinContent( ipx, iex, -1. );

    }
  }

  for ( unsigned int i=0; i<clients_.size(); i++ ) {

    EEIntegrityClient* ebit = dynamic_cast<EEIntegrityClient*>(clients_[i]);
    if ( ebit ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        MonitorElement* me = ebit->meg01_[ism-1];

        if ( me ) {

          for ( int ie = 1; ie <= 85; ie++ ) {
            for ( int ip = 1; ip <= 20; ip++ ) {

              float xval = me->getBinContent( ie, ip );

              int iex;
              int ipx;

              if ( ism <= 9 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-10);
              }

              meIntegrity_->setBinContent( ipx, iex, xval );

            }
          }

        }

      }

    }

    EEPedestalOnlineClient* ebpo = dynamic_cast<EEPedestalOnlineClient*>(clients_[i]);
    if ( ebpo ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        MonitorElement* me = ebpo->meg03_[ism-1];

        if ( me ) {

          for ( int ie = 1; ie <= 85; ie++ ) {
            for ( int ip = 1; ip <= 20; ip++ ) {

              float xval = me->getBinContent( ie, ip );

              int iex;
              int ipx;

              if ( ism <= 9 ) {
                iex = 1+(85-ie);
                ipx = ip+20*(ism-1);
              } else {
                iex = 85+ie;
                ipx = 1+(20-ip)+20*(ism-10);
              }

              mePedestalOnline_->setBinContent( ipx, iex, xval );

            }
          }

        }

      }

    }

  }

}

void EESummaryClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EESummaryClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 400;

//  const double histMax = 1.e15;

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  // dummy histogram labelling the SM's
  TH2C labelGrid("labelGrid","label grid for SM", 9, 0., 360., 2, -85., 85.);
  for ( short sm=0; sm<18; sm++ ) {
    int x = 1 + sm%9;
    int y = 1 + sm/9;
    labelGrid.SetBinContent(x, y, Numbers::iEE(sm+1));
  }
  labelGrid.SetMarkerSize(2);
  labelGrid.SetMinimum(-9.01);

  string imgNameMapI, imgNameMapPO, imgName, meName;

  TCanvas* cMap = new TCanvas("cMap", "Temp", int(360./170.*csize), csize);

  float saveHeigth = gStyle->GetTitleH();
  gStyle->SetTitleH(0.07);
  float saveFontSize = gStyle->GetTitleFontSize();
  gStyle->SetTitleFontSize(15);

  TH2F* obj2f;

  imgNameMapI = "";

  gStyle->SetPaintTextFormat("+g");

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( meIntegrity_ );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapI = meName + ".png";
    imgName = htmlDir + imgNameMapI;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(9, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->SetTitleSize(0.5);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  imgNameMapPO = "";

  obj2f = 0;
  obj2f = UtilsClient::getHisto<TH2F*>( mePedestalOnline_ );

  if ( obj2f ) {

    meName = obj2f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1 ,"_" );
      }
    }
    imgNameMapPO = meName + ".png";
    imgName = htmlDir + imgNameMapPO;

    cMap->cd();
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(6, pCol3);
    obj2f->GetXaxis()->SetNdivisions(9, kFALSE);
    obj2f->GetYaxis()->SetNdivisions(2);
    cMap->SetGridx();
    cMap->SetGridy();
    obj2f->SetMinimum(-0.00000001);
    obj2f->SetMaximum(6.0);
    obj2f->Draw("col");
    labelGrid.Draw("text,same");
    cMap->Update();
    cMap->SaveAs(imgName.c_str());

  }

  gStyle->SetPaintTextFormat();

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapI.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapI << "\" usemap=""#Integrity"" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;

  if ( imgNameMapPO.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameMapPO << "\" usemap=""#PedestalOnline"" border=0></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  delete cMap;

  gStyle->SetPaintTextFormat();

  if ( imgNameMapI.size() != 0 ) this->writeMap( htmlFile, "Integrity" );
  if ( imgNameMapPO.size() != 0 ) this->writeMap( htmlFile, "PedestalOnline" );

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  gStyle->SetTitleH( saveHeigth );
  gStyle->SetTitleFontSize( saveFontSize );

}

void EESummaryClient::writeMap( std::ofstream& hf, std::string mapname ) {

 std::map<std::string, std::string> refhtml;
 refhtml["Integrity"] = "EEIntegrityClient.html";
 refhtml["PedestalOnline"] = "EEPedestalOnlineClient.html";

 const int A0 =  85;
 const int A1 = 759;
 const int B0 =  35;
 const int B1 = 334;

 hf << "<map name=\"" << mapname << "\">" << std::endl;
 for( unsigned int sm=0; sm<superModules_.size(); sm++ ) {
  int i=(superModules_[sm]-1)/9;
  int j=(superModules_[sm]-1)%9;
  int x0 = A0 + (A1-A0)*j/9;
  int x1 = A0 + (A1-A0)*(j+1)/9;
  int y0 = B0 + (B1-B0)*(1-i)/2;
  int y1 = B0 + (B1-B0)*((1-i)+1)/2;
  hf << "<area title=\"" << Numbers::sEE((j+1)+9*i).c_str()
     << "\" shape=\"rect\" href=\"" << refhtml[mapname] << "#"
     << Numbers::sEE((j+1)+9*i).c_str() << "\" coords=\"";
  hf << x0 << ", " << y0 << ", " << x1 << ", " << y1 << "\">" << std::endl;
 }
 hf << "</map>" << std::endl;

}

