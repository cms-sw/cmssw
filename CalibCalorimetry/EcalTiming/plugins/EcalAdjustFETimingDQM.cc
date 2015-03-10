// -*- C++ -*-
//
// Package:    EcalAdjustFETimingDQM
// Class:      EcalAdjustFETimingDQM
// 
/**\class EcalAdjustFETimingDQM EcalAdjustFETimingDQM.cc CalibCalorimetry/EcalTiming/plugins/EcalAdjustFETimingDQM.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Seth Cooper,27 1-024,+41227672342,
//         Created:  Mon Sep 26 17:38:06 CEST 2011
// $Id: EcalAdjustFETimingDQM.cc,v 1.12 2012/09/27 15:37:43 scooper Exp $
// Second Author: Tambe E. Norbert
//                Univ Of Minnesota
//
//
// ***************************************************************************************
// Program to create FE time adjustments from DQM and apply them to current settings
// Creates output root file with timing histograms and XML files with the FE/TT delays.
// 

#include "DQM/EcalCommon/interface/Numbers.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/EcalTiming/plugins/EcalAdjustFETimingDQM.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "TFile.h"
#include "TH1.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <memory>

const int numEEsm     = 18;
const int maxNumCCUinFed = EcalTrigTowerDetId::kEBTowersPerSM+2;

//
// constructors and destructor
//
EcalAdjustFETimingDQM::EcalAdjustFETimingDQM(const edm::ParameterSet& iConfig) :
  ebDQMFileName_ (iConfig.getParameter<std::string>("EBDQMFileName")),
  eeDQMFileName_ (iConfig.getParameter<std::string>("EEDQMFileName")),
  xmlFileNameBeg_ (iConfig.getParameter<std::string>("XMLFileNameBeg")),
  txtFileName_ (iConfig.getParameter<std::string>("TextFileName")),
  rootFileNameBeg_ (iConfig.getParameter<std::string>("RootFileNameBeg")),
  readDelaysFromDB_ (iConfig.getParameter<bool>("ReadExistingDelaysFromDB")),
  minTimeChangeToApply_	(iConfig.getParameter<double>("MinTimeChangeToApply")),
  operateInDumpMode_	(iConfig.getParameter<bool>("OperateInDumpMode"))
{
  if(operateInDumpMode_) std::cout << "++ the program EcalAdjustFETimingDQM will operate in dump mode: fill xml files with valus in conf database, w/o modifications" << std::endl;
  else                   std::cout << "++ the program EcalAdjustFETimingDQM will operate in normal mode: fill xml files with valus in conf database after the modifications from inout DQM file" << std::endl;
}


EcalAdjustFETimingDQM::~EcalAdjustFETimingDQM()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
EcalAdjustFETimingDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;

  int runNum = getRunNumber(ebDQMFileName_);
  if(runNum <= 0)
  {
    cout << "Error: Unusual root file name. Unable to parse run number." << endl;
    return;
  }
  int runNumEE = getRunNumber(eeDQMFileName_);
  if(runNumEE != runNum)
  {
    cout << "Error: Run numbers in the names of the two root files given are mismatched: "
      << runNum << " vs. " << runNumEE << endl;
    return;
  }

  cout << "Run number: " << runNum << endl;

  TFile* fileEB = TFile::Open(ebDQMFileName_.c_str());
  TFile* fileEE = TFile::Open(eeDQMFileName_.c_str());
  // Look in fileEB/fileEE and try to get the timing maps
  string path = "DQMData/Run ";
  path+=intToString(runNum);
  string pathEB=path;
  string pathEE=path;
  pathEB+="/EcalBarrel/Run summary/EBTimingTask/";
  pathEE+="/EcalEndcap/Run summary/EETimingTask/";;
  TProfile2D* ttMapEBorig = (TProfile2D*) fileEB->Get((pathEB+"EBTMT timing map").c_str());
  TProfile2D* ttMapEEPorig = (TProfile2D*) fileEE->Get((pathEE+"EETMT timing map EE +").c_str());
  TProfile2D* ttMapEEMorig = (TProfile2D*) fileEE->Get((pathEE+"EETMT timing map EE -").c_str());
  if(!ttMapEBorig)
  {
    cout << "Error: EB timing map not found in first input file, " << ebDQMFileName_ << endl;
    return;
  }
  if(!ttMapEEPorig || !ttMapEEMorig)
  {
    cout << "Error: EE timing maps not found in second input file, " << eeDQMFileName_ << endl;
    return;
  }

  // We should have good maps at this point
  string runNumstr = intToString(runNum);
  string filename = rootFileNameBeg_;
  filename+=runNumstr;
  filename+=".root";
  TFile* output = new TFile(filename.c_str(),"recreate");
  output->cd();
  TH1F* timingTTrunEBHist = new TH1F("timingRunTTEB","trigger tower timing EB;ns",100,-5,5);
  TH1F* timingTTrunEEPHist = new TH1F("timingRunTTEEP","trigger tower timing EEP;ns",100,-5,5);
  TH1F* timingTTrunEEMHist = new TH1F("timingRunTTEEM","trigger tower timing EEM;ns",100,-5,5);
  TH1F* adjTTrunEBHist = new TH1F("adjustmentTTEB","adjustment to trigger tower timing EB;ns",100,-5,5);
  TH1F* adjTTrunEEPHist = new TH1F("adjustmentTTEEP","adjustment to trigger tower timing EEP;ns",100,-5,5);
  TH1F* adjTTrunEEMHist = new TH1F("adjustmentTTEEM","adjustment to trigger tower timing EEM;ns",100,-5,5);
  TProfile2D* ttMapEB =  (TProfile2D*) ttMapEBorig->Clone();
  TProfile2D* ttMapEEP = (TProfile2D*) ttMapEEPorig->Clone();
  TProfile2D* ttMapEEM = (TProfile2D*) ttMapEEMorig->Clone();
  // rename originals
  string ebMapstr = "TT Timing EB Run ";
  ebMapstr+=runNumstr;
  ttMapEB->SetNameTitle(ebMapstr.c_str(),ebMapstr.c_str());
  timingTTrunEBHist->SetTitle(ebMapstr.c_str());
  string eepMapstr = "TT Timing EE+ Run ";
  eepMapstr+=runNumstr;
  ttMapEEP->SetNameTitle(eepMapstr.c_str(),eepMapstr.c_str());
  timingTTrunEEPHist->SetTitle(eepMapstr.c_str());
  string eemMapstr = "TT Timing EE- Run ";
  eemMapstr+=runNumstr;
  ttMapEEM->SetNameTitle(eemMapstr.c_str(),eemMapstr.c_str());
  timingTTrunEEMHist->SetTitle(eepMapstr.c_str());

  // initialize geometry and array of tt times
  Numbers::initGeometry(iSetup,true);
  float ttAvgTimesEB[EBDetId::MAX_SM][70];
  int ttNumEntriesEB[EBDetId::MAX_SM][70];
  float ttAvgTimesEE[numEEsm][maxNumCCUinFed];
  int ttNumEntriesEE[numEEsm][maxNumCCUinFed];
  for(int i=0; i<EBDetId::MAX_SM; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      ttAvgTimesEB[i][j] = 0;
      ttNumEntriesEB[i][j] = 0;
    }
  }
  for(int i=0; i<numEEsm; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      ttAvgTimesEE[i][j] = 0;
      ttNumEntriesEE[i][j] = 0;
    }
  }

  // loop over TT map, fill array and 1-D hist -- EB
  for(int xphi=1; xphi < 73; ++xphi)
  {
    for(int yeta=1; yeta < 35; ++yeta)
    {
      int ietaAbs = (yeta < 18) ? 18-yeta : yeta-17;
      int iphi = xphi;
      int zside = (yeta < 18) ? -1 : 1;
      int centerCryIeta = zside*(ietaAbs*5 - 2);
      int centerCryIphi = iphi*5 - 2;
      EBDetId det(centerCryIeta,centerCryIphi);
      // which SM?
      int iSM = Numbers::iSM(det);
      // which TT/SC?
      int iTT = Numbers::iSC(iSM,EcalBarrel,det.ietaSM(),det.iphiSM());

      if(ttMapEB->GetBinEntries(ttMapEB->GetBin(xphi,yeta)) > 0)
      {
        ttAvgTimesEB[iSM-1][iTT-1]+= ttMapEB->GetBinContent(xphi,yeta);
        ttNumEntriesEB[iSM-1][iTT-1]++;
      }

      ////debug
      //if(iSM==36)
      //{
      //  cout << "xphi: " << xphi << " yeta: " << yeta
      //    << " ietaCenter: " << det.ietaSM() << " iPhiCenter: " << det.iphiSM()
      //    << " iSM: " << iSM << " sEB: " << Numbers::sEB(iSM)
      //    << " iTT: " << iTT << " binc: " << ttMapEB->GetBinContent(xphi,yeta)
      //    << endl;
      //}
    }
  }
  // create avg tower times
  for(int i=0; i<EBDetId::MAX_SM; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j) // add two to account for mem boxes
    {
      if(ttNumEntriesEB[i][j]>0) 
        timingTTrunEBHist->Fill(ttAvgTimesEB[i][j]/ttNumEntriesEB[i][j]);
      // choice of sign according to:
      // http://cmsonline.cern.ch/portal/page/portal/CMS%20online%20system/Elog?_piref815_429145_815_429142_429142.strutsAction=%2FviewMessageDetails.do%3FmsgId%3D667090
      // 24 hardware counts correspond to 25 ns => rescale averages by 24./25.
      if( fabs( ttAvgTimesEB[i][j]/ttNumEntriesEB[i][j] ) > minTimeChangeToApply_ ) 
      {
        ttAvgTimesEB[i][j] = floor(ttAvgTimesEB[i][j]/ttNumEntriesEB[i][j]*24./25+0.5);
      }
      else
      {
        ttAvgTimesEB[i][j] = 0;
      }
      adjTTrunEBHist->Fill(ttAvgTimesEB[i][j]);
      ////debug
      //if(i==35)
      //{
      //  cout << "AvgTime for iTT/iSC " << j+1 << " in iSM 36 = "
      //    << Numbers::sEB(36) << " : "
      //    << ttAvgTimesEB[35][j] << endl;
      //}
    }
  }

  // loop over TT map, fill array and 1-D hist -- EE
  for(int x=1; x < 20; ++x)
  {
    for(int y=1; y < 20; ++y)
    {
      if(ttMapEEM->GetBinEntries(ttMapEEM->GetBin(x,y)) > 0)
      {
        int centerCryIx = x*5 - 1;
        int centerCryIy = y*5 - 1;
        // special cases for two inside corner TT's
        if(y==9 && x==12)
          centerCryIy-=3;
        else if(y==12 && x==9)
          centerCryIx-=3;

        EEDetId detMinus;
        while(detMinus==EEDetId() && centerCryIy >= (y-1)*5 && centerCryIx >= (x-1)*5)
        {
          try
          {
            detMinus = EEDetId(centerCryIx,centerCryIy,-1);
          } catch(exception& e) {
            centerCryIy--;
            centerCryIx--;
          }
        }
        if(detMinus==EEDetId())
        {
          cout << "Considering x=" << x << " y=" << y << endl;
          cout << "ERROR: unable to find crystal in this bin!" << endl;
          cout << "Final coordinates; x=" << centerCryIx << " y=" << centerCryIy << endl;
          return;
        }

        // which SM?
        int iSMminus = Numbers::iSM(detMinus);
        // which TT/SC?
        int iSCminus = Numbers::iSC(iSMminus,EcalEndcap,detMinus.ix(),detMinus.iy());
        ttAvgTimesEE[iSMminus-1][iSCminus-1]+= ttMapEEM->GetBinContent(x,y);
        ttNumEntriesEE[iSMminus-1][iSCminus-1]++;
        ////debug
        //if(iSMminus==8)
        //{
        //  cout << "x: " << x << " y: " << y
        //    << " ixCenter: " << detMinus.ix() << " iyCenter: " << detMinus.iy()
        //    << " iSM: " << iSMminus << " sEE: " << Numbers::sEE(iSMminus)
        //    << " iSCminus: " << iSCminus << " binc: " << ttMapEEM->GetBinContent(x,y)
        //    << endl;
        //}
      }


      if(ttMapEEP->GetBinEntries(ttMapEEP->GetBin(x,y)) > 0)
      {
        int centerCryIx = x*5 - 1;
        int centerCryIy = y*5 - 1;
        // special cases for two inside corner TT's
        if(y==9 && x==12)
          centerCryIy-=3;
        else if(y==12 && x==9)
          centerCryIx-=3;
        EEDetId detPlus;
        while(detPlus==EEDetId() && centerCryIy >= (y-1)*5 && centerCryIx >= (x-1)*5)
        {
          try
          {
            detPlus = EEDetId(centerCryIx,centerCryIy,1);
          } catch(exception& e) {
            centerCryIy--;
            centerCryIx--;
          }
        }
        if(detPlus==EEDetId())
        {
          cout << "Considering x=" << x << " y=" << y << endl;
          cout << "ERROR: unable to find crystal in this bin!" << endl;
          cout << "Final coordinates; x=" << centerCryIx << " y=" << centerCryIy << endl;
          return;
        }

        // which SM?
        int iSMplus = Numbers::iSM(detPlus);
        // which TT/SC?
        int iSCplus = Numbers::iSC(iSMplus,EcalEndcap,detPlus.ix(),detPlus.iy());
        ttAvgTimesEE[iSMplus-1][iSCplus-1]+= ttMapEEP->GetBinContent(x,y);
        ttNumEntriesEE[iSMplus-1][iSCplus-1]++;
        ////debug
        //if(iSMplus==11)
        //{
        //  cout << "x: " << x << " y: " << y
        //    << " ixCenter: " << detPlus.ix() << " iyCenter: " << detPlus.iy()
        //    << " iSM: " << iSMplus << " sEE: " << Numbers::sEE(iSMplus)
        //    << " iSCplus: " << iSCplus << " binc: " << ttMapEEP->GetBinContent(x,y)
        //    << endl;
        //}
      }

    }
  }

  // create avg tower times -- EE
  for(int i=0; i<numEEsm; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      if(i<=8 && ttNumEntriesEE[i][j]>0)
        timingTTrunEEMHist->Fill(ttAvgTimesEE[i][j]/ttNumEntriesEE[i][j]);
      else if(ttNumEntriesEE[i][j]>0)
        timingTTrunEEPHist->Fill(ttAvgTimesEE[i][j]/ttNumEntriesEE[i][j]);

      // choice of sign according to:
      // http://cmsonline.cern.ch/portal/page/portal/CMS%20online%20system/Elog?_piref815_429145_815_429142_429142.strutsAction=%2FviewMessageDetails.do%3FmsgId%3D667090
      // 24 hardware counts correspond to 25 ns => rescale averages by 24./25.
      if( fabs( ttAvgTimesEE[i][j]/ttNumEntriesEE[i][j] ) > minTimeChangeToApply_ ) 
      {
        ttAvgTimesEE[i][j] = floor(ttAvgTimesEE[i][j]/ttNumEntriesEE[i][j]*24./25+0.5);
      }
      else
      {
        ttAvgTimesEE[i][j] = 0;
      }
      if(i<=8)
        adjTTrunEEMHist->Fill(ttAvgTimesEE[i][j]);
      else
        adjTTrunEEPHist->Fill(ttAvgTimesEE[i][j]);

      ////debug
      //if(i==10)
      //{
      //  cout << "AvgTime for iTT/iSC " << j+1 << " in iSM 11 = "
      //    << Numbers::sEE(11) << " : "
      //    << ttAvgTimesEE[10][j]
      //    << " entries: " << ttNumEntriesEE[10][j] << endl;
      //}
    }
  }

  // now read the DB and get the absolute delays
  int feDelaysFromDBEB[EBDetId::MAX_SM][maxNumCCUinFed];
  int feDelaysFromDBEE[numEEsm][maxNumCCUinFed];
  for(int i=0; i<EBDetId::MAX_SM; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      feDelaysFromDBEB[i][j] = -999999;
    }
  }
  for(int i=0; i<numEEsm; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      feDelaysFromDBEE[i][j] = -999999;
    }
  }

  if(readDelaysFromDB_)
  {
    string sid;
    string user;
    string pass;
    std::cout << "Please enter sid: ";
    std::cin >> sid;
    std::cout << "Please enter user: ";
    std::cin >> user;
    std::cout << "Please enter password: ";
    std::cin >> pass;
    EcalCondDBInterface* econn;
    try {
      try {
        cout << "Making DB connection..." << flush;
        econn = new EcalCondDBInterface( sid, user, pass );
        cout << "Done." << endl;
      } catch (runtime_error &e) {
        cerr << e.what() << endl;
        exit(-1);
      }
      RunIOV iov = econn->fetchRunIOV("P5_Co", runNum);
      std::list<ODDelaysDat> delays = econn->fetchFEDelaysForRun(&iov);
      std::list<ODDelaysDat>::const_iterator i = delays.begin();
      std::list<ODDelaysDat>::const_iterator e = delays.end();
      while (i != e)
      {
        int idcc = i->getFedId()-600;
        int ism = 0;
        if(idcc >= 1 && idcc <= 9)        // EEM
          ism = idcc;
        else if(idcc >= 10 && idcc <= 45) // EB
          ism = idcc-9;
        else if(idcc >= 46 && idcc <= 54) // EEP
          ism = idcc-45+9;
        else
          std::cout << "warning: strange iDCC read from db: " << idcc << ". " << std::endl;

        int ccuId = i->getTTId();
        if(idcc >= 10 && idcc <= 45) // EB
        {
          if(feDelaysFromDBEB[ism-1][ccuId-1] != -999999)
            std::cout << "warning: duplicate entry in DB found for fed: " << idcc+600
              << " CCU: " << ccuId << "; replacing old entry with this one." << std::endl;
          feDelaysFromDBEB[ism-1][ccuId-1] = i->getTimeOffset();
        }
        else if( (idcc >= 1 && idcc <= 9) || (idcc >= 46 && idcc <= 54)) // EE
        {
          if(feDelaysFromDBEE[ism-1][ccuId-1] != -999999)
            std::cout << "warning: duplicate entry in DB found for fed: " << idcc+600
              << " CCU: " << ccuId << "; replacing old entry with this one." << std::endl;
          feDelaysFromDBEE[ism-1][ccuId-1] = i->getTimeOffset();
        }

        i++;
      }
    } catch (exception &e) {
      cout << "ERROR:  " << e.what() << endl;
    } catch (...) {
      cout << "Unknown error caught" << endl;
    }
    std::cout << "ECAL hardware latency settings retrieved from db for run: " << runNum << std::endl; 
  }

  // now we should have the filled DB array
  // loop over the old ttAvgTimes arrays and adjust
  // recall that we have +1*avgTT time in the ttAvgTimes arrays (i.e., the needed shift)
  // so here just add that to the DB delays
  // make new arrays for absolute time
  // one 24 hardware counts correspond to 25 ns => rescale averages by 24./25.
  int newFEDelaysEB[EBDetId::MAX_SM][maxNumCCUinFed];
  int newFEDelaysEE[numEEsm][maxNumCCUinFed];
  for(int i=0; i<EBDetId::MAX_SM; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      if (!operateInDumpMode_) newFEDelaysEB[i][j] = ttAvgTimesEB[i][j] + feDelaysFromDBEB[i][j];
      else                     newFEDelaysEB[i][j] = feDelaysFromDBEB[i][j];
    }
  }
  // create avg tower times -- EE
  for(int i=0; i<numEEsm; ++i)
  {
    for(int j=0; j<maxNumCCUinFed; ++j)
    {
      if (!operateInDumpMode_) newFEDelaysEE[i][j] = ttAvgTimesEE[i][j] + feDelaysFromDBEE[i][j];
      else                     newFEDelaysEE[i][j] =  feDelaysFromDBEE[i][j];
    }
  }

  /////////////////////////////////////////////////////////////	
  // write output
  ofstream txt_outfile;
  txt_outfile.open(txtFileName_.c_str(),ios::out);
  txt_outfile << "#  Needed shift in terms of samples and fine tuning (ns) for each TT"
    << endl;
  txt_outfile << "#   shift" << std::endl;

  // EB
  for(int ism=1; ism<=EBDetId::MAX_SM; ++ism)
  {
    for(int iTT=1;iTT<69;++iTT)   // ignoring two mem boxes ONLY for the text file
    {
      txt_outfile << 609+ism << setw(6) << Numbers::sEB(ism) <<setw(4) << iTT << "  " << setw(4)
        << ttAvgTimesEB[ism-1][iTT-1] << "\t" << endl;  
      if(fabs(ttAvgTimesEB[ism-1][iTT-1]) > 1)
        cout << "WARNING: Unusually large shift found!  SM=" << 609+ism
          << " " << Numbers::sEB(ism) << " iTT=" << iTT
        << " shift: " << ttAvgTimesEB[ism-1][iTT-1] << endl;  
    }

    // XMLs
    ofstream xml_outfile;
    string xmlFileName = xmlFileNameBeg_;
    xmlFileName+=intToString(609+ism);
    xmlFileName+=".xml";
    xml_outfile.open(xmlFileName.c_str(),ios::out);

    xml_outfile << "<delayOffsets>" << endl;
    xml_outfile << " <DELAY_OFFSET_RELEASE VERSION_ID = \"SM" << 609+ism <<" _VER1\"> \n";
    xml_outfile << "      <RELEASE_ID>RELEASE_1</RELEASE_ID>\n";
    xml_outfile << "             <SUPERMODULE>" << 609+ism << "</SUPERMODULE>\n";
    xml_outfile << "     <TIME_STAMP> 270705 </TIME_STAMP>" << endl;

    for(int j=68; j<maxNumCCUinFed; ++j)
    {
      if(feDelaysFromDBEB[ism-1][j] == -999999 ) continue;     // if db did not give this CCU at the start, don't out put it
      xml_outfile << "   <DELAY_OFFSET>\n";
      xml_outfile << "             <SUPERMODULE>" << 609+ism <<"</SUPERMODULE>\n";
      xml_outfile << "             <TRIGGERTOWER>" << j+1 << "</TRIGGERTOWER>\n";
      xml_outfile << "             <TIME_OFFSET>" << newFEDelaysEB[ism-1][j] << "</TIME_OFFSET>\n";
      xml_outfile << "    </DELAY_OFFSET>" << endl;
    }
    for(int j=0; j<(maxNumCCUinFed-2); ++j)
    {
      if(feDelaysFromDBEB[ism-1][j] == -999999 ) continue;     // if db did not give this CCU at the start, don't out put it
      xml_outfile << "   <DELAY_OFFSET>\n";
      xml_outfile << "             <SUPERMODULE>" << 609+ism <<"</SUPERMODULE>\n";
      xml_outfile << "             <TRIGGERTOWER>" << j+1 << "</TRIGGERTOWER>\n";
      xml_outfile << "             <TIME_OFFSET>" << newFEDelaysEB[ism-1][j] << "</TIME_OFFSET>\n";
      xml_outfile << "    </DELAY_OFFSET>" << endl;
    }
    xml_outfile << " </DELAY_OFFSET_RELEASE>" << endl;
    xml_outfile << "</delayOffsets>" << endl;
    xml_outfile.close();
  }

  // EE
  for(int ism=1; ism<=numEEsm; ++ism)
  {
    int iDCC = ism<=9 ? ism : ism + 45 - 9;

    for(int iSC=0; iSC<68; ++iSC)   // ignoring two mem boxes ONLY for the text file
    {
      if(feDelaysFromDBEE[ism-1][iSC] == -999999 ) continue;     // if db did not give this CCU at the start, don't out put it
      txt_outfile << 600+iDCC << setw(6) << Numbers::sEE(ism) <<setw(4) << (iSC+1) << "  " << setw(4)
        << ttAvgTimesEE[ism-1][iSC] << "\t" << endl;  
      if(fabs(ttAvgTimesEE[ism-1][iSC]) > 1)
        cout << "WARNING: Unusually large shift found!  SM=" << 600+iDCC
          << " " << Numbers::sEE(ism) << " iSC=" << (iSC+1)
          << " shift: " << ttAvgTimesEE[ism-1][iSC] << endl;
    }

    // XMLs
    ofstream xml_outfile;
    string xmlFileName = xmlFileNameBeg_;
    xmlFileName+=intToString(600+iDCC);
    xmlFileName+=".xml";
    xml_outfile.open(xmlFileName.c_str(),ios::out);

    xml_outfile << "<delayOffsets>" << endl;
    xml_outfile << " <DELAY_OFFSET_RELEASE VERSION_ID = \"SM" << 600+iDCC <<" _VER1\"> \n";
    xml_outfile << "      <RELEASE_ID>RELEASE_1</RELEASE_ID>\n";
    xml_outfile << "             <SUPERMODULE>" << 600+iDCC << "</SUPERMODULE>\n";
    xml_outfile << "     <TIME_STAMP> 270705 </TIME_STAMP>" << endl;

    for(int j=68; j<maxNumCCUinFed; ++j)
    {
      if(feDelaysFromDBEE[ism-1][j] == -999999 ) continue;     // if db did not give this CCU at the start, don't out put it
      xml_outfile << "   <DELAY_OFFSET>\n";
      xml_outfile << "             <SUPERMODULE>" << 600+iDCC <<"</SUPERMODULE>\n";
      xml_outfile << "             <TRIGGERTOWER>" << j+1 << "</TRIGGERTOWER>\n";
      xml_outfile << "             <TIME_OFFSET>" << newFEDelaysEE[ism-1][j] << "</TIME_OFFSET>\n";
      xml_outfile << "    </DELAY_OFFSET>" << endl;
    }
    for(int j=0; j<68; ++j)
    {
      if(feDelaysFromDBEE[ism-1][j] == -999999 ) continue;     // if db did not give this CCU at the start, don't out put it
      xml_outfile << "   <DELAY_OFFSET>\n";
      xml_outfile << "             <SUPERMODULE>" << 600+iDCC <<"</SUPERMODULE>\n";
      xml_outfile << "             <TRIGGERTOWER>" << j+1 << "</TRIGGERTOWER>\n";
      xml_outfile << "             <TIME_OFFSET>" << newFEDelaysEE[ism-1][j] << "</TIME_OFFSET>\n";
      xml_outfile << "    </DELAY_OFFSET>" << endl;
    }
    xml_outfile << " </DELAY_OFFSET_RELEASE>" << endl;
    xml_outfile << "</delayOffsets>" << endl;
    xml_outfile.close();
  }


  txt_outfile.close();

  // scale original maps
  for(int xBin = 0; xBin < ttMapEB->GetNbinsX()+1; ++xBin)
  {
    for(int yBin = 0; yBin < ttMapEB->GetNbinsY()+1; ++yBin)
    {
      ttMapEB->SetBinContent(xBin,yBin,ttMapEB->GetBinContent(xBin,yBin)-50);
      ttMapEB->SetBinEntries(ttMapEB->GetBin(xBin,yBin),1);
    }
  }
  for(int xBin = 0; xBin < ttMapEEM->GetNbinsX()+1; ++xBin)
  {
    for(int yBin = 0; yBin < ttMapEEM->GetNbinsY()+1; ++yBin)
    {
      ttMapEEM->SetBinContent(xBin,yBin,ttMapEEM->GetBinContent(xBin,yBin)-50);
      ttMapEEM->SetBinEntries(ttMapEEM->GetBin(xBin,yBin),1);
    }
  }
  for(int xBin = 0; xBin < ttMapEEP->GetNbinsX()+1; ++xBin)
  {
    for(int yBin = 0; yBin < ttMapEEP->GetNbinsY()+1; ++yBin)
    {
      ttMapEEP->SetBinContent(xBin,yBin,ttMapEEP->GetBinContent(xBin,yBin)-50);
      ttMapEEP->SetBinEntries(ttMapEEP->GetBin(xBin,yBin),1);
    }
  }
  // Move bins without entries away from zero and set range
  moveBinsTProfile2D(ttMapEB);
  moveBinsTProfile2D(ttMapEEP);
  moveBinsTProfile2D(ttMapEEM);
  ttMapEB->SetMinimum(-2);
  ttMapEEP->SetMinimum(-2);
  ttMapEEM->SetMinimum(-2);
  ttMapEB->SetMaximum(2);
  ttMapEEP->SetMaximum(2);
  ttMapEEM->SetMaximum(2);

  // Write
  ttMapEB->Write();
  ttMapEEP->Write();
  ttMapEEM->Write();
  timingTTrunEBHist->Write();
  timingTTrunEEPHist->Write();
  timingTTrunEEMHist->Write();
  adjTTrunEBHist->Write();
  adjTTrunEEMHist->Write();
  adjTTrunEEPHist->Write();


  output->Close();
}


// ------------ method called once each job just before starting event loop  ------------
void 
EcalAdjustFETimingDQM::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalAdjustFETimingDQM::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
EcalAdjustFETimingDQM::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
EcalAdjustFETimingDQM::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
EcalAdjustFETimingDQM::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
EcalAdjustFETimingDQM::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EcalAdjustFETimingDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void EcalAdjustFETimingDQM::moveBinsTProfile2D(TProfile2D* myprof)
{
  int nxbins = myprof->GetNbinsX();
  int nybins = myprof->GetNbinsY();

  for(int i=0; i<=(nxbins+2)*(nybins+2); i++ )
  {
    Double_t binents = myprof->GetBinEntries(i);
    if(binents == 0)
    {
      myprof->SetBinEntries(i,1);
      myprof->SetBinContent(i,-1000);
    }
  }
  return;
}

void EcalAdjustFETimingDQM::scaleBinsTProfile2D(TProfile2D* myprof, double weight)
{
  int nxbins = myprof->GetNbinsX();
  int nybins = myprof->GetNbinsY();

  for(int i=0; i<=(nxbins+2)*(nybins+2); i++ )
    myprof->SetBinContent(i,myprof->GetBinContent(i)+weight);

  return;
}

std::string EcalAdjustFETimingDQM::intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str());
}

// Fish run number out of file name
int EcalAdjustFETimingDQM::getRunNumber(std::string fileName)
{
  using namespace std;

  int runNumPos = fileName.find(".root");
  int Rpos = fileName.find("_R");
  if(runNumPos <= 0 || Rpos <= 0)
    return -1;

  string runNumString = fileName.substr(Rpos+2,runNumPos-Rpos-2);
  stringstream convert(runNumString);
  int runNumber;
  if(!(convert >> runNumber))
    runNumber = -1;
  return runNumber;
}

