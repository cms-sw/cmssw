/*
 * \file EETriggerTowerClient.cc
 *
 * $Date: 2007/11/10 16:09:25 $
 * $Revision: 1.22 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <bitset>

#include "TStyle.h"
#include "TGraph.h"
#include "TLine.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/LogicID.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETriggerTowerClient::EETriggerTowerClient(const ParameterSet& ps){

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  // prefix to ME paths
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // vector of selected Super Modules (Defaults to all 18).
  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    i01_[ism-1] = 0;
    j01_[ism-1] = 0;
    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    h02_[ism-1] = 0;
    i02_[ism-1] = 0;
    j02_[ism-1] = 0;


    meh01_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

    meh02_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej02_[ism-1] = 0;


//     for (int j = 0; j < 34 ; j++) {

//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;

//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;

//     }

  }

}

EETriggerTowerClient::~EETriggerTowerClient(){

}

void EETriggerTowerClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;
  dbe_ = mui->getBEInterface();

  if ( verbose_ ) cout << "EETriggerTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EETriggerTowerClient::beginRun(void) {

  if ( verbose_ ) cout << "EETriggerTowerClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EETriggerTowerClient::endJob(void) {

  if ( verbose_ ) cout << "EETriggerTowerClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETriggerTowerClient::endRun(void) {

  if ( verbose_ ) cout << "EETriggerTowerClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETriggerTowerClient::setup(void) {

}

void EETriggerTowerClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( l01_[ism-1] ) delete l01_[ism-1];
      if ( m01_[ism-1] ) delete m01_[ism-1];
      if ( n01_[ism-1] ) delete n01_[ism-1];
    }

    h01_[ism-1] = 0;
    i01_[ism-1] = 0;
    j01_[ism-1] = 0;

    h02_[ism-1] = 0;
    i02_[ism-1] = 0;
    j02_[ism-1] = 0;
    l01_[ism-1] = 0;
    m01_[ism-1] = 0;
    n01_[ism-1] = 0;

    meh01_[ism-1] = 0;
    mei01_[ism-1] = 0;
    mej01_[ism-1] = 0;
    mel01_[ism-1] = 0;
    mem01_[ism-1] = 0;
    men01_[ism-1] = 0;

    meh02_[ism-1] = 0;
    mei02_[ism-1] = 0;
    mej02_[ism-1] = 0;


//     for ( int j = 0; j < 34 ; j++ ) {

//       if ( cloneME_ ) {
//         if ( k01_[ism-1][j] ) delete k01_[ism-1][j];
//         if ( k02_[ism-1][j] ) delete k02_[ism-1][j];
//       }

//       k01_[ism-1][j] = 0;
//       k02_[ism-1][j] = 0;

//       mek01_[ism-1][j] = 0;
//       mek02_[ism-1][j] = 0;

//     }

  }

}

bool EETriggerTowerClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  return status;

}

void EETriggerTowerClient::subscribe(void){

  if ( verbose_ ) cout << "EETriggerTowerClient: subscribe" << endl;

  subscribe( "Real Digis",
             "EETriggerTowerTask", false);

  subscribe( "Emulated Digis",
             "EETriggerTowerTask/Emulated", true);

}

void EETriggerTowerClient::subscribe( const char* nameext,
                                      const char* folder,
                                      bool emulated ) {

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/%s/EETTT Et map %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT FineGrainVeto %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT Flags %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFlagError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->subscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFineGrainVetoError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->subscribe(histo, ism);
    }

//     for (int j = 0; j < 34 ; j++) {
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//     }

  }

}

void EETriggerTowerClient::subscribeNew(void){

  subscribeNew( "Real Digis",
                "EETriggerTowerTask", false );

  subscribeNew( "Emulated Digis",
                "EETriggerTowerTask/Emulated", true );

}

void EETriggerTowerClient::subscribeNew( const char* nameext,
                                         const char* folder,
                                         bool emulated ) {
  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/%s/EETTT Et map %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT FineGrainVeto %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT Flags %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFlagError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->subscribeNew(histo, ism);
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFineGrainVetoError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->subscribeNew(histo, ism);
    }

//     for (int j = 0; j < 34 ; j++) {
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribeNew(histo, ism);
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribeNew(histo, ism);
//     }

  }

}

void EETriggerTowerClient::unsubscribe(void){

  if ( verbose_ ) cout << "EETriggerTowerClient: unsubscribe" << endl;

  unsubscribe( "Real Digis",
               "EETriggerTowerTask", false );

  unsubscribe( "Emulated Digis",
               "EETriggerTowerTask/Emulated", true);

}

void EETriggerTowerClient::unsubscribe( const char* nameext,
                                        const char* folder,
                                        bool emulated ) {

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/%s/EETTT Et map %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT FineGrainVeto %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);
    sprintf(histo, "*/EcalEndcap/%s/EETTT Flags %s %s", folder, nameext, Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

    if(!emulated) {
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->unsubscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFlagError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->unsubscribe(histo, ism);
      sprintf(histo, "*/EcalEndcap/%s/EETTT EmulFineGrainVetoError %s %s", folder, nameext, Numbers::sEE(ism).c_str());
      mui_->unsubscribe(histo, ism);
    }

//     for (int j = 0; j < 34 ; j++) {
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//       sprintf(histo, "*/EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R %s TT%02d", ism, j+1);
//       mui_->subscribe(histo, ism);
//     }

  }

}

void EETriggerTowerClient::softReset(void){

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meh01_[ism-1] ) dbe_->softReset(meh01_[ism-1]);
    if ( mei01_[ism-1] ) dbe_->softReset(mei01_[ism-1]);
    if ( mej01_[ism-1] ) dbe_->softReset(mej01_[ism-1]);
    if ( mel01_[ism-1] ) dbe_->softReset(mel01_[ism-1]);
    if ( mem01_[ism-1] ) dbe_->softReset(mem01_[ism-1]);
    if ( men01_[ism-1] ) dbe_->softReset(men01_[ism-1]);
    if ( meh02_[ism-1] ) dbe_->softReset(meh02_[ism-1]);
    if ( mei02_[ism-1] ) dbe_->softReset(mei02_[ism-1]);
    if ( mej02_[ism-1] ) dbe_->softReset(mej02_[ism-1]);

//     for (int j = 0; j < 34 ; j++) {

//       if ( mek01_[ism-1][j] ) dbe_->softReset(mek01_[ism-1][j]);
//       if ( mek02_[ism-1][j] ) dbe_->softReset(mek02_[ism-1][j]);

//     }

  }

}

void EETriggerTowerClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EETriggerTowerClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  analyze("Real Digis",
          "EETriggerTowerTask", false );

  analyze("Emulated Digis",
          "EETriggerTowerTask/Emulated", true );

}

void EETriggerTowerClient::analyze(const char* nameext,
                                   const char* folder,
                                   bool emulated) {
  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT Et map %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      h01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h01_[ism-1] );
      if(h01_[ism-1]) h01_[ism-1]->SetEntries(1.+h01_[ism-1]->GetEntries());
      meh01_[ism-1] = me;
    }
    else {
      h02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, h02_[ism-1] );
      if(h02_[ism-1]) h02_[ism-1]->SetEntries(1.+h02_[ism-1]->GetEntries());
      meh02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT FineGrainVeto %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      i01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i01_[ism-1] );
      if(i01_[ism-1]) i01_[ism-1]->SetEntries(1.+i01_[ism-1]->GetEntries());
      mei01_[ism-1] = me;
    }
    else {
      i02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, i02_[ism-1] );
      if(i02_[ism-1]) i02_[ism-1]->SetEntries(1.+i02_[ism-1]->GetEntries());
      mei02_[ism-1] = me;
    }

    sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT Flags %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
    me = dbe_->get(histo);
    if(!emulated) {
      j01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j01_[ism-1] );
      if(j01_[ism-1]) j01_[ism-1]->SetEntries(1.+j01_[ism-1]->GetEntries());
      mej01_[ism-1] = me;
    }
    else {
      j02_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, j02_[ism-1] );
      if(j02_[ism-1]) j02_[ism-1]->SetEntries(1.+j02_[ism-1]->GetEntries());
      mej02_[ism-1] = me;
    }

    if(!emulated) {
      sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT EmulError %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
      me = dbe_->get(histo);
      l01_[ism-1] = UtilsClient::getHisto<TH2F*>( me, cloneME_, l01_[ism-1] );
      if(l01_[ism-1]) l01_[ism-1]->SetEntries(1.+l01_[ism-1]->GetEntries());
      mel01_[ism-1] = me;

      sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT EmulFlagError %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
      me = dbe_->get(histo);
      m01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, m01_[ism-1] );
      if(m01_[ism-1]) m01_[ism-1]->SetEntries(1.+m01_[ism-1]->GetEntries());
      mem01_[ism-1] = me;

      sprintf(histo, (prefixME_+"EcalEndcap/%s/EETTT EmulFineGrainVetoError %s %s").c_str(), folder, nameext, Numbers::sEE(ism).c_str());
      me = dbe_->get(histo);
      n01_[ism-1] = UtilsClient::getHisto<TH3F*>( me, cloneME_, n01_[ism-1] );
      if(n01_[ism-1]) n01_[ism-1]->SetEntries(1.+n01_[ism-1]->GetEntries());
      men01_[ism-1] = me;

    }

//     for (int j = 0; j < 34 ; j++) {

//       sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et T %s TT%02d").c_str(), ism, j+1);
//       me = dbe_->get(histo);
//       k01_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k01_[ism-1][j] );
//       mek01_[ism-1][j] = me;

//       sprintf(histo, (prefixME_+"EcalEndcap/EETriggerTowerTask/EnergyMaps/EETTT Et R %s TT%02d").c_str(), ism, j+1);
//       me = dbe_->get(histo);
//       k02_[ism-1][j] = UtilsClient::getHisto<TH1F*>( me, cloneME_, k02_[ism-1][j] );
//       mek02_[ism-1][j] = me;

//     }

  }

}

void EETriggerTowerClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EETriggerTowerClient html output ..." << std::endl;

  std::ofstream htmlFile[19];

  htmlFile[0].open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile[0] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile[0] << "<html>  " << std::endl;
  htmlFile[0] << "<head>  " << std::endl;
  htmlFile[0] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile[0] << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile[0] << "  <title>Monitor:TriggerTowerTask output</title> " << std::endl;
  htmlFile[0] << "</head>  " << std::endl;
  htmlFile[0] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile[0] << "<body>  " << std::endl;
  //htmlFile[0] << "<br>  " << std::endl;
  htmlFile[0] << "<a name=""top""></a>" << endl;
  htmlFile[0] << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile[0] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << std::endl;
  htmlFile[0] << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile[0] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h2> " << std::endl;
  htmlFile[0] << "<br>" << std::endl;
  //htmlFile[0] << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=lime>channel has NO problems</td>" << std::endl;
  //htmlFile[0] << "<td bgcolor=yellow>channel is missing</td></table>" << std::endl;
  htmlFile[0] << "<hr>" << std::endl;
  htmlFile[0] << "<table border=1>" << std::endl;
  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {
    htmlFile[0] << "<td bgcolor=white><a href=""#"
                << Numbers::sEE(superModules_[i]).c_str() << ">"
                << setfill( '0' ) << setw(2) << superModules_[i] << "</a></td>";
  }
  htmlFile[0] << std::endl << "</table>" << std::endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  //const double histMax = 1.e15;

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

  string imgName[3], meName[3], imgMeName[3];

  TCanvas* cMe1 = new TCanvas("cMe1", "Temp", 2*csize, 2*csize);
  //  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.2*csize), int(1.2*csize));
  //  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.4*csize), int(0.4*csize));
  TCanvas* cMe2 = new TCanvas("cMe2", "Temp", int(1.8*csize), int(1.8*csize));
  TCanvas* cMe3 = new TCanvas("cMe3", "Temp", int(0.9*csize), int(0.9*csize));

  TH2F* obj2f;
  TH3F* obj3f;
  TProfile2D* obj2p;

  // Loop on endcap supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    if ( i>0 ) htmlFile[0] << "<a href=""#top"">Top</a>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;
    htmlFile[0] << "<h3><a name="""
                << Numbers::sEE(ism).c_str() << """></a><strong>"
                << Numbers::sEE(ism).c_str() << "</strong></h3>" << endl;


    // ---------------------------  Emulator Error

    htmlFile[0] << "<h3><strong>Emulator Error</strong></h3>" << std::endl;
    htmlFile[0] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[0] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[0] << "<tr align=\"center\">" << std::endl;


    imgName[0] = "";

    obj2f = l01_[ism-1];

    if ( obj2f ) {

      meName[0] = obj2f->GetName();

      for ( unsigned int i = 0; i < meName[0].size(); i++ ) {
        if ( meName[0].substr(i, 1) == " " )  {
          meName[0].replace(i, 1 ,"_" );
        }
      }

      imgName[0] = meName[0] + ".png";
      imgMeName[0] = htmlDir + imgName[0];

      cMe2->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(10, pCol4);
      obj2f->SetMinimum(0);
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetXaxis()->SetTitleSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetTitleSize(0.02);
      obj2f->GetZaxis()->SetLabelSize(0.02);
      cMe2->SetGridx();
      cMe2->SetGridy();
      obj2f->Draw("colz");
      int x1 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+0.);
      int x2 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+50.);
      int y1 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+0.);
      int y2 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+50.);
      labelGrid.GetXaxis()->SetRange(x1, x2);
      labelGrid.GetYaxis()->SetRange(y1, y2);
      labelGrid.Draw("text,same");
      cMe2->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cMe2->Update();
      cMe2->SaveAs(imgMeName[0].c_str());
    }


    htmlFile[0] << "<img src=\"" << imgName[0] << "\"><br>" << std::endl;
    htmlFile[0] << "</tr>" << std::endl;
    htmlFile[0] << "<br><br>" << std::endl;


    // ---------------------------  Et plots

    for(int iemu=0;iemu<2;iemu++) {

      imgName[iemu] = "";

      obj3f = (iemu+1==1) ? h01_[ism-1] : h02_[ism-1];

      if ( obj3f ) {
        meName[iemu] = obj3f->GetName();

        for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
          if ( meName[iemu].substr(i, 1) == " " )  {
            meName[iemu].replace(i, 1 ,"_" );
          }
        }

        imgName[iemu] = meName[iemu] + ".png";
        imgMeName[iemu] = htmlDir + imgName[iemu];

        obj2p = obj3f->Project3DProfile("yx");

        cMe1->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);

        std::string projname(obj2p->GetName());
        string::size_type loc = projname.find( "_pyx", 0 );
        projname.replace( loc, projname.length(), "");
        obj2p->SetTitle(projname.c_str());

        cMe1->SetGridx();
        cMe1->SetGridy();
        obj2p->GetXaxis()->SetLabelSize(0.02);
        obj2p->GetXaxis()->SetTitleSize(0.02);
        obj2p->GetYaxis()->SetLabelSize(0.02);
        obj2p->GetYaxis()->SetTitleSize(0.02);
        obj2p->GetZaxis()->SetLabelSize(0.02);
        obj2p->Draw("colz");
        int x1 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+0.);
        int x2 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+50.);
        int y1 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+0.);
        int y2 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+50.);
        labelGrid.GetXaxis()->SetRange(x1, x2);
        labelGrid.GetYaxis()->SetRange(y1, y2);
        labelGrid.Draw("text,same");
        cMe1->SetBit(TGraph::kClipFrame);
        TLine l;
        l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
          if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
            l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
          }
        }
        cMe1->Update();
        cMe1->SaveAs(imgMeName[iemu].c_str());
        delete obj2p;
      }
    }

    htmlFile[0] << "<td><img src=\"" << imgName[0] << "\"></td>" << std::endl;
    htmlFile[0] << "<td><img src=\"" << imgName[1] << "\"></td>" << std::endl;
    htmlFile[0] << "</table>" << std::endl;
    htmlFile[0] << "<br>" << std::endl;

    std::stringstream subpage;
    subpage << htmlName.substr( 0, htmlName.find( ".html" ) ) << "_" << Numbers::sEE(ism).c_str() << ".html" << std::ends;
    htmlFile[0] << "<a href=\"" << subpage.str().c_str() << "\">" << Numbers::sEE(ism).c_str() << " details</a><br>" << std::endl;
    htmlFile[0] << "<hr>" << std::endl;

    htmlFile[ism].open((htmlDir + subpage.str()).c_str());

    // html page header
    htmlFile[ism] << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
    htmlFile[ism] << "<html>  " << std::endl;
    htmlFile[ism] << "<head>  " << std::endl;
    htmlFile[ism] << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
    htmlFile[ism] << " http-equiv=\"content-type\">  " << std::endl;
    htmlFile[ism] << "  <title>Monitor:TriggerTowerTask output " << Numbers::sEE(ism).c_str()  << "</title> " << std::endl;
    htmlFile[ism] << "</head>  " << std::endl;
    htmlFile[ism] << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
    htmlFile[ism] << "<body>  " << std::endl;
    htmlFile[ism] << "<br>  " << std::endl;
    htmlFile[ism] << "<h3>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h3>" << std::endl;
    htmlFile[ism] << "<h3>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">TRIGGER TOWER</span></h3> " << std::endl;
    htmlFile[ism] << "<h3>SM:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
    htmlFile[ism] << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
    htmlFile[ism] << " style=\"color: rgb(0, 0, 153);\">" << Numbers::sEE(ism).c_str() << "</span></h3>" << std::endl;
    htmlFile[ism] << "<hr>" << std::endl;

    // ---------------------------  Flag bits plots

    htmlFile[ism] << "<h3><strong>Trigger Tower Flags</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    int counter = 0;

    for ( int j=1; j<8; j++ ) {

      for(int iemu=0;iemu<3;iemu++) {

        imgName[iemu] = "";

        if( iemu==0 ) obj3f = m01_[ism-1];
        else if( iemu==1 ) obj3f = j01_[ism-1];
        else if( iemu==2 ) obj3f = j02_[ism-1];

        if ( obj3f ) {

          meName[iemu] = obj3f->GetName();

          for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
            if ( meName[iemu].substr(i, 1) == " " )  {
              meName[iemu].replace(i, 1 ,"_" );
            }
          }

          if ( j == 3 ) continue;   //  010 bits combination is not used
          counter++;
          if ( j < 7 ) {
            imgName[iemu] = meName[iemu] + "_" + char(47+j) + ".png";
            obj3f->GetZaxis()->SetRange( j, j );
          }
          else {
            imgName[iemu] = meName[iemu] + "_6-7.png";
            obj3f->GetZaxis()->SetRange( j, j+1 );
          }
          imgMeName[iemu] = htmlDir + imgName[iemu];

          obj2f = (TH2F*) obj3f->Project3D( "yx" );

          cMe2->cd();
          gStyle->SetOptStat(" ");
          gStyle->SetPalette(10, pCol4);
          obj2f->SetMinimum(0);
          obj2f->GetXaxis()->SetLabelSize(0.02);
          obj2f->GetXaxis()->SetTitleSize(0.02);
          obj2f->GetYaxis()->SetLabelSize(0.02);
          obj2f->GetYaxis()->SetTitleSize(0.02);
          obj2f->GetZaxis()->SetLabelSize(0.02);
          cMe2->SetGridx();
          cMe2->SetGridy();

          std::stringstream title;
          std::string emustring;
          if (iemu==0) emustring = "Errors ";
          else if(iemu==1) emustring = "Real Digis ";
          else if(iemu==2) emustring = "Emulated Digis ";
          if ( j < 7 ) {
            title << "EETTT Flags " << emustring << " Bit " << bitset<3>(j-1) << Numbers::sEE(ism).c_str();
          } else {
            title << "EETTT Flags " << emustring << " Bits 110+111 " << Numbers::sEE(ism).c_str();
          }
          obj2f->SetTitle( title.str().c_str() );

          obj2f->Draw("colz");
          int x1 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+0.);
          int x2 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+50.);
          int y1 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+0.);
          int y2 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+50.);
          labelGrid.GetXaxis()->SetRange(x1, x2);
          labelGrid.GetYaxis()->SetRange(y1, y2);
          labelGrid.Draw("text,same");
          cMe2->SetBit(TGraph::kClipFrame);
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
            }
          }
          cMe2->Update();
          cMe2->SaveAs(imgMeName[iemu].c_str());

          delete obj2f;

          htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;
          if ( counter%3 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;
        }
      }
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;



    // ---------------------------  Fine Grain Veto

    htmlFile[ism] << "<h3><strong>Fine Grain Veto</strong></h3>" << std::endl;
    htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    htmlFile[ism] << "<tr align=\"center\">" << std::endl;


    for ( int j=1; j<=2; j++ ) {

      for(int iemu=0;iemu<3;iemu++) {

        imgName[iemu] = "";

        if( iemu==0 ) obj3f = n01_[ism-1];
        else if( iemu==1 ) obj3f = i01_[ism-1];
        else if( iemu==2 ) obj3f = i02_[ism-1];

        if ( obj3f ) {

          meName[iemu] = obj3f->GetName();

          for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
            if ( meName[iemu].substr(i, 1) == " " )  {
              meName[iemu].replace(i, 1 ,"_" );
            }
          }

          imgName[iemu] = meName[iemu] + "_" + char(47+j) + ".png";
          imgMeName[iemu] = htmlDir + imgName[iemu];

          obj3f->GetZaxis()->SetRange( j, j );

          obj2f = (TH2F*) obj3f->Project3D( "yx" );

          cMe2->cd();
          gStyle->SetOptStat(" ");
          gStyle->SetPalette(10, pCol4);
          obj2f->SetMinimum(0);
          obj2f->GetXaxis()->SetLabelSize(0.02);
          obj2f->GetXaxis()->SetTitleSize(0.02);
          obj2f->GetYaxis()->SetLabelSize(0.02);
          obj2f->GetYaxis()->SetTitleSize(0.02);
          obj2f->GetZaxis()->SetLabelSize(0.02);
          cMe2->SetGridx();
          cMe2->SetGridy();

          std::stringstream title;
          std::string emustring;
          if (iemu==0) emustring = "Errors ";
          else if(iemu==1) emustring = "Real Digis ";
          else if(iemu==2) emustring = "Emulated Digis ";
          title << "EETTT FineGrainVeto " << emustring << Numbers::sEE(ism).c_str() << ", FineGrainVeto = " << j-1;
          obj2f->SetTitle( title.str().c_str() );

          obj2f->Draw("colz");
          int x1 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+0.);
          int x2 = labelGrid.GetXaxis()->FindBin(Numbers::ix0EE(ism)+50.);
          int y1 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+0.);
          int y2 = labelGrid.GetYaxis()->FindBin(Numbers::iy0EE(ism)+50.);
          labelGrid.GetXaxis()->SetRange(x1, x2);
          labelGrid.GetYaxis()->SetRange(y1, y2);
          labelGrid.Draw("text,same");
          cMe2->SetBit(TGraph::kClipFrame);
          TLine l;
          l.SetLineWidth(1);
          for ( int i=0; i<201; i=i+1){
            if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
              l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
            }
          }
          cMe2->Update();
          cMe2->SaveAs(imgMeName[iemu].c_str());
          delete obj2f;

          htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;
        }
      }
      htmlFile[ism] << "</tr><tr>" << std::endl;
    }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // ---------------------------  Et plots per Tower

    //     htmlFile[ism] << "<h3><strong>Et</strong></h3>" << std::endl;
    //     htmlFile[ism] << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
    //     htmlFile[ism] << "cellpadding=\"10\" align=\"center\"> " << std::endl;
    //     htmlFile[ism] << "<tr align=\"center\">" << std::endl;

    //     for ( int j=0; j<34; j++ ) {

    //       TH1F* obj1f1 = k01_[ism-1][j];
    //       TH1F* obj1f2 = k02_[ism-1][j];

    //       if ( obj1f1 ) {

    //         imgName[iemu] = "";

    //         meName[iemu] = obj1f1->GetName();

    //         for ( unsigned int i = 0; i < meName[iemu].size(); i++ ) {
    //           if ( meName[iemu].substr(i, 1) == " " )  {
    //             meName[iemu].replace(i, 1 ,"_" );
    //           }
    //         }

    //         imgName[iemu] = meName[iemu] + ".png";
    //         imgMeName[iemu] = htmlDir + imgName[iemu];

    //         cMe3->cd();
    //         gStyle->SetOptStat("euomr");
    //         if ( obj1f2 ) {
    //           float m = TMath::Max( obj1f1->GetMaximum(), obj1f2->GetMaximum() );
    //           obj1f1->SetMaximum( m + 1. );
    //         }
    //         obj1f1->SetStats(kTRUE);
    //         gStyle->SetStatW( gStyle->GetStatW() * 1.5 );
    //         obj1f1->Draw();
    //         cMe3->Update();

    //         if ( obj1f2 ) {
    //           gStyle->SetStatY( gStyle->GetStatY() - 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kRed );
    //           obj1f2->SetStats(kTRUE);
    //           obj1f2->SetLineColor( kRed );
    //           obj1f2->Draw( "sames" );
    //           cMe3->Update();
    //           gStyle->SetStatY( gStyle->GetStatY() + 1.25*gStyle->GetStatH() );
    //           gStyle->SetStatTextColor( kBlack );
    //         }

    //         gStyle->SetStatW( gStyle->GetStatW() / 1.5 );
    //         cMe3->SaveAs(imgMeName[iemu].c_str());

    //         htmlFile[ism] << "<td><img src=\"" << imgName[iemu] << "\"></td>" << std::endl;

    //       }

    //       if ( (j+1)%4 == 0 ) htmlFile[ism] << "</tr><tr>" << std::endl;

    //     }

    htmlFile[ism] << "</tr>" << std::endl << "</table>" << std::endl;

    // html page footer
    htmlFile[ism] << "</body> " << std::endl;
    htmlFile[ism] << "</html> " << std::endl;
    htmlFile[ism].close();
  }

  delete cMe1;
  delete cMe2;
  delete cMe3;

  // html page footer
  htmlFile[0] << "</body> " << std::endl;
  htmlFile[0] << "</html> " << std::endl;
  htmlFile[0].close();

}

