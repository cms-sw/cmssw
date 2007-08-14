/*
 * \file EETimingClient.cc
 *
 * $Date: 2007/08/09 14:36:55 $
 * $Revision: 1.13 $
 * \author G. Della Ricca
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
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorClient/interface/EETimingClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EETimingClient::EETimingClient(const ParameterSet& ps){

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

    meh01_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;

    mea01_[ism-1] = 0;

    mep01_[ism-1] = 0;

    mer01_[ism-1] = 0;

    qth01_[ism-1] = 0;

    qtg01_[ism-1] = 0;

  }

  expectedMean_ = 6.0;
  discrepancyMean_ = 0.5;
  RMSThreshold_ = 0.5;

}

EETimingClient::~EETimingClient(){

}

void EETimingClient::beginJob(MonitorUserInterface* mui){

  mui_ = mui;

  if ( verbose_ ) cout << "EETimingClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  if ( enableQT_ ) {

    Char_t qtname[200];

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(qtname, "EETMT quality %s", Numbers::sEE(ism).c_str());
      qth01_[ism-1] = dynamic_cast<MEContentsProf2DWithinRangeROOT*> (mui_->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(), qtname));

      qth01_[ism-1]->setMeanRange(expectedMean_ - discrepancyMean_, expectedMean_ + discrepancyMean_);

      qth01_[ism-1]->setRMSRange(0.0, RMSThreshold_);

      qth01_[ism-1]->setMinimumEntries(10*1700);

      qth01_[ism-1]->setErrorProb(1.00);

      sprintf(qtname, "EETMT quality test %s", Numbers::sEE(ism).c_str());
      qtg01_[ism-1] = dynamic_cast<MEContentsTH2FWithinRangeROOT*> (mui_->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(), qtname));

      qtg01_[ism-1]->setMeanRange(1., 6.);

      qtg01_[ism-1]->setErrorProb(1.00);

    }

  }

}

void EETimingClient::beginRun(void){

  if ( verbose_ ) cout << "EETimingClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EETimingClient::endJob(void) {

  if ( verbose_ ) cout << "EETimingClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETimingClient::endRun(void) {

  if ( verbose_ ) cout << "EETimingClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EETimingClient::setup(void) {

  Char_t histo[200];

  mui_->setCurrentFolder( "EcalEndcap/EETimingClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    sprintf(histo, "EETMT timing quality %s", Numbers::sEE(ism).c_str());
    meg01_[ism-1] = dbe->book2D(histo, histo, 50, Numbers::ix0EE(ism)+0., Numbers::ix0EE(ism)+50., 50, Numbers::iy0EE(ism)+0., Numbers::iy0EE(ism)+50.);

    if ( mea01_[ism-1] ) dbe->removeElement( mea01_[ism-1]->getName() );
    sprintf(histo, "EETMT timing %s", Numbers::sEE(ism).c_str());
    mea01_[ism-1] = dbe->book1D(histo, histo, 1700, 0., 1700.);

    if ( mep01_[ism-1] ) dbe->removeElement( mep01_[ism-1]->getName() );
    sprintf(histo, "EETMT timing mean %s", Numbers::sEE(ism).c_str());
    mep01_[ism-1] = dbe->book1D(histo, histo, 100, 0.0, 10.0);

    if ( mer01_[ism-1] ) dbe->removeElement( mer01_[ism-1]->getName() );
    sprintf(histo, "EETMT timing rms %s", Numbers::sEE(ism).c_str());
    mer01_[ism-1] = dbe->book1D(histo, histo, 100, 0.0,  2.5);

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    UtilsClient::resetHisto( meg01_[ism-1] );

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        meg01_[ism-1]->setBinContent( ix, iy, -1. );

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( Numbers::validEE(ism, jx, jy) ) {
          meg01_[ism-1]->setBinContent( ix, iy, 2. );
        }

      }
    }

    UtilsClient::resetHisto( mea01_[ism-1] );
    UtilsClient::resetHisto( mep01_[ism-1] );
    UtilsClient::resetHisto( mer01_[ism-1] );

  }

}

void EETimingClient::cleanup(void) {

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
    }

    h01_[ism-1] = 0;

    meh01_[ism-1] = 0;

  }

  mui_->setCurrentFolder( "EcalEndcap/EETimingClient" );
  DaqMonitorBEInterface* dbe = mui_->getBEInterface();

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) dbe->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;

    if ( mea01_[ism-1] ) dbe->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;

    if ( mep01_[ism-1] ) dbe->removeElement( mep01_[ism-1]->getName() );
    mep01_[ism-1] = 0;

    if ( mer01_[ism-1] ) dbe->removeElement( mer01_[ism-1]->getName() );
    mer01_[ism-1] = 0;

  }

}

bool EETimingClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) {

  bool status = true;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    cout << " SM=" << ism << endl;

    UtilsClient::printBadChannels(qth01_[ism-1]);

//    UtilsClient::printBadChannels(qtg01_[ism-1]);

  }

  return status;

}

void EETimingClient::subscribe(void){

  if ( verbose_ ) cout << "EETimingClient: subscribe" << endl;

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
    mui_->subscribe(histo, ism);

  }

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETimingClient: collate" << endl;

    for ( unsigned int i=0; i<superModules_.size(); i++ ) {

      int ism = superModules_[i];

      sprintf(histo, "EETMT timing %s", Numbers::sEE(ism).c_str());
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalEndcap/Sums/EETimingTask");
      sprintf(histo, "*/EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
      mui_->add(me_h01_[ism-1], histo);

    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
      if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
    } else {
      if ( enableMonitorDaemon_ ) {
        sprintf(histo, "*/EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      } else {
        sprintf(histo, "EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
        if ( qth01_[ism-1] ) mui_->useQTest(histo, qth01_[ism-1]->getName());
      }
    }

    sprintf(histo, "EcalEndcap/EETimingClient/EETMT timing quality %s", Numbers::sEE(ism).c_str());
    if ( qtg01_[ism-1] ) mui_->useQTest(histo, qtg01_[ism-1]->getName());

  }

}

void EETimingClient::subscribeNew(void){

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
    mui_->subscribeNew(histo, ism);

  }

}

void EETimingClient::unsubscribe(void){

  if ( verbose_ ) cout << "EETimingClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EETimingClient: uncollate" << endl;

    if ( mui_ ) {

      for ( unsigned int i=0; i<superModules_.size(); i++ ) {

        int ism = superModules_[i];

        mui_->removeCollate(me_h01_[ism-1]);

      }

    }

  }

  Char_t histo[200];

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    unsigned int ism = superModules_[i];

    sprintf(histo, "*/EcalEndcap/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
    mui_->unsubscribe(histo, ism);

  }

}

void EETimingClient::softReset(void){

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meh01_[ism-1] ) mui_->softReset(meh01_[ism-1]);

  }

}

void EETimingClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EETimingClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("TIMING_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("TIMING_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("TIMING_MEAN_ERROR");
  bits01 |= EcalErrorDictionary::getMask("TIMING_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask;

  EcalErrorMask::fetchDataSet(&mask);

  Char_t histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( collateSources_ ) {
      sprintf(histo, "EcalEndcap/Sums/EETimingTask/EETMT timing %s", Numbers::sEE(ism).c_str());
    } else {
      sprintf(histo, (prefixME_+"EcalEndcap/EETimingTask/EETMT timing %s").c_str(), Numbers::sEE(ism).c_str());
    }
    me = mui_->get(histo);
    h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );
    meh01_[ism-1] = me;

    UtilsClient::resetHisto( meg01_[ism-1] );
    UtilsClient::resetHisto( mea01_[ism-1] );
    UtilsClient::resetHisto( mep01_[ism-1] );
    UtilsClient::resetHisto( mer01_[ism-1] );

    for ( int ix = 1; ix <= 50; ix++ ) {
      for ( int iy = 1; iy <= 50; iy++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, -1.);

        int jx = ix + Numbers::ix0EE(ism);
        int jy = iy + Numbers::iy0EE(ism);

        if ( Numbers::validEE(ism, jx, jy) ) {
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ix, iy, 2. );
        }

        bool update01;

        float num01;
        float mean01;
        float rms01;

        update01 = UtilsClient::getBinStats(h01_[ism-1], ix, iy, num01, mean01, rms01);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - expectedMean_) > discrepancyMean_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(ix, iy, val);

          if ( mea01_[ism-1] ) {
            if ( mean01 > 0. ) {
              mea01_[ism-1]->setBinContent(iy+20*(ix-1), mean01);
              mea01_[ism-1]->setBinError(iy+20*(ix-1), rms01);
            } else {
              mea01_[ism-1]->setEntries(1.+mea01_[ism-1]->getEntries());
            }
          }
          if ( mep01_[ism-1] ) mep01_[ism-1]->Fill(mean01);
          if ( mer01_[ism-1] ) mer01_[ism-1]->Fill(rms01);

        }

        // masking

        if ( mask.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask.begin(); m != mask.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = Numbers::icEE(ism, ix, iy);

            if ( ic == -1 ) continue;

            if ( ecid.getID1() == Numbers::iSM(ism, EcalEndcap) && ecid.getID2() == ic ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                if ( meg01_[ism-1] ) {
                  float val = int(meg01_[ism-1]->getBinContent(ix, iy)) % 3;
                  meg01_[ism-1]->setBinContent( ix, iy, val+3 );
                }
              }
            }

          }
        }

      }
    }

    vector<dqm::me_util::Channel> badChannels;

    if ( qth01_[ism-1] ) badChannels = qth01_[ism-1]->getBadChannels();

//    if ( ! badChannels.empty() ) {
//      for ( vector<dqm::me_util::Channel>::iterator it = badChannels.begin(); it != badChannels.end(); ++it ) {
//        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent(it->getBinX(), it->getBinY(), 0.);
//      }
//    }

  }

}

void EETimingClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EETimingClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:TimingTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TIMING</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;
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

  int pCol3[6] = { 301, 302, 303, 304, 305, 306 };

  string imgNameQual, imgNameTim, imgNameMean, imgNameRMS, imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 0, 0, 2*csize, 2*csize);
  TCanvas* cTim = new TCanvas("cTim", "Temp", 0, 0, csize, csize);
  TCanvas* cMean = new TCanvas("cMean", "Temp", 0, 0, csize, csize);
  TCanvas* cRMS = new TCanvas("cRMS", "Temp", 0, 0, csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;

  // Loop on barrel supermodules

  for ( unsigned int i=0; i<superModules_.size(); i ++ ) {

    int ism = superModules_[i];

    // Quality plots

    imgNameQual = "";

    obj2f = 0;
    obj2f = UtilsClient::getHisto<TH2F*>( meg01_[ism-1] );

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameQual = meName + ".png";
      imgName = htmlDir + imgNameQual;

      cQual->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(6, pCol3);
      cQual->SetGridx();
      cQual->SetGridy();
      obj2f->SetMinimum(-0.00000001);
      obj2f->SetMaximum(6.0);
      obj2f->GetXaxis()->SetLabelSize(0.02);
      obj2f->GetYaxis()->SetLabelSize(0.02);
      obj2f->Draw("col");
      cQual->SetBit(TGraph::kClipFrame);
      TLine l;
      l.SetLineWidth(1);
      for ( int i=0; i<201; i=i+1){
        if ( (Numbers::ixSectorsEE[i]!=0 || Numbers::iySectorsEE[i]!=0) && (Numbers::ixSectorsEE[i+1]!=0 || Numbers::iySectorsEE[i+1]!=0) ) {
          l.DrawLine(Numbers::ixSectorsEE[i], Numbers::iySectorsEE[i], Numbers::ixSectorsEE[i+1], Numbers::iySectorsEE[i+1]);
        }
      }
      cQual->Update();
      cQual->SaveAs(imgName.c_str());

    }

    // Timing distributions

    imgNameTim = "";

    obj1f = 0;
    obj1f = UtilsClient::getHisto<TH1F*>( mea01_[ism-1] );

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameTim = meName + ".png";
      imgName = htmlDir + imgNameTim;

      cTim->cd();
      gStyle->SetOptStat("euo");
      obj1f->SetStats(kTRUE);
//      if ( obj1f->GetMaximum(histMax) > 0. ) {
//        gPad->SetLogy(1);
//      } else {
//        gPad->SetLogy(0);
//      }
      obj1f->SetMinimum(0.0);
      obj1f->SetMaximum(10.0);
      obj1f->Draw();
      cTim->Update();
      cTim->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    // Mean distributions

    imgNameMean = "";

    obj1f = 0;
    obj1f = UtilsClient::getHisto<TH1F*>( mep01_[ism-1] );

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameMean = meName + ".png";
      imgName = htmlDir + imgNameMean;

      cMean->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cMean->Update();
      cMean->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    // RMS distributions

    obj1f = 0;
    obj1f = UtilsClient::getHisto<TH1F*>( mer01_[ism-1] );

    imgNameRMS = "";

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameRMS = meName + ".png";
      imgName = htmlDir + imgNameRMS;

      cRMS->cd();
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cRMS->Update();
      cRMS->SaveAs(imgName.c_str());
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

    if ( imgNameQual.size() != 0 )
      htmlFile << "<td colspan=\"3\"><img src=\"" << imgNameQual << "\"></td>" << endl;
    else
      htmlFile << "<td colspan=\"3\"><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    if ( imgNameTim.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameTim << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameMean.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameMean << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameRMS.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameRMS << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cTim;
  delete cMean;
  delete cRMS;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

