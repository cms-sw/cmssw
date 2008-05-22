/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/01 00:39:51 $
 *  $Revision: 1.15 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTLocalTriggerTest::DTLocalTriggerTest(const edm::ParameterSet& ps){

  setConfig(ps,"DTLocalTrigger");

}


DTLocalTriggerTest::~DTLocalTriggerTest(){

}


void DTLocalTriggerTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  edm::LogVerbatim ("localTrigger") <<"[" << testName << "Test]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("localTrigger") <<"[" << testName << "Test]: "<<nLumiSegs<<" updates";
  

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      for (int stat=1; stat<=4; ++stat){
	for (int wh=-2; wh<=2; ++wh){
	  for (int sect=1; sect<=12; ++sect){
	    DTChamberId chId(wh,stat,sect);
	    int sector_id = (wh+3)+(sect-1)*5;
	    // uint32_t indexCh = chId.rawId();
	    

	    // Perform DCC/DDU common plot analysis (Phi ones)
	    TH2F * BXvsQual      = getHisto<TH2F>(dbe->get(getMEName("BXvsQual","LocalTriggerPhi", chId)));
	    TH1F * BestQual      = getHisto<TH1F>(dbe->get(getMEName("BestQual","LocalTriggerPhi", chId)));
	    TH2F * Flag1stvsQual = getHisto<TH2F>(dbe->get(getMEName("Flag1stvsQual","LocalTriggerPhi", chId)));
	    if (BXvsQual && Flag1stvsQual && BestQual) {
	      
	      TH1D* BXHH    = BXvsQual->ProjectionY("",7,7,"");
	      TH1D* Flag1st = Flag1stvsQual->ProjectionY();
	      int BXOK_bin = BXHH->GetEntries()>=1 ? BXHH->GetMaximumBin() : 51;
	      double BX_OK =  BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
	      double trigsFlag2nd = Flag1st->GetBinContent(2);
	      double trigs = Flag1st->GetEntries();
	      double besttrigs = BestQual->GetEntries();
	      double besttrigsCorr = 0;
	      for (int i=5;i<=7;++i)
		besttrigsCorr+=BestQual->GetBinContent(i);
	      
	      if( secME[sector_id].find(fullName("CorrectBXSecPhi")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrectBXPhi");
		bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrFractionPhi");
		bookSectorHistos(wh,sect,"LocalTriggerPhi","2ndFractionPhi");
	      }
	      if( whME[wh].find(fullName("CorrectBXPhi")) == whME[wh].end() ){
		bookWheelHistos(wh,"LocalTriggerPhi","CorrectBXPhi");
		bookWheelHistos(wh,"LocalTriggerPhi","CorrFractionPhi");
		bookWheelHistos(wh,"LocalTriggerPhi","2ndFractionPhi");
		bookWheelHistos(wh,"LocalTriggerPhi","TriggerInclusivePhi");
	      }
	      std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
	      innerME->find(fullName("CorrectBXPhi"))->second->setBinContent(stat,BX_OK);
	      innerME->find(fullName("CorrFractionPhi"))->second->setBinContent(stat,besttrigsCorr/besttrigs);
	      innerME->find(fullName("2ndFractionPhi"))->second->setBinContent(stat,trigsFlag2nd/trigs);
	   
	      innerME = &(whME[wh]);
	      innerME->find(fullName("CorrectBXPhi"))->second->setBinContent(sect,stat,BX_OK);
	      innerME->find(fullName("CorrFractionPhi"))->second->setBinContent(sect,stat,besttrigsCorr/besttrigs);
	      innerME->find(fullName("TriggerInclusivePhi"))->second->setBinContent(sect,stat,besttrigs);
	      innerME->find(fullName("2ndFractionPhi"))->second->setBinContent(sect,stat,trigsFlag2nd/trigs);
	    
	    }

// 	    // Perform analysis on DCC exclusive plots (Phi)	  
// 	    TH2F * QualvsPhirad  = getHisto<TH2F>(dbe->get(getMEName("QualvsPhirad","LocalTriggerPhi", chId)));
// 	    TH2F * QualvsPhibend = getHisto<TH2F>(dbe->get(getMEName("QualvsPhibend","LocalTriggerPhi", chId)));
// 	    if (QualvsPhirad && QualvsPhibend) {
	      
// 	      TH1D* phiR = QualvsPhirad->ProjectionX();
// 	      TH1D* phiB = QualvsPhibend->ProjectionX("_px",5,7,"");

// 	      if( chambME[indexCh].find(fullName("TrigDirectionPhi")) == chambME[indexCh].end() ){
// 		bookChambHistos(chId,"TrigDirectionPhi");
// 		bookChambHistos(chId,"TrigPositionPhi");
// 	      }
// 	      std::map<std::string,MonitorElement*> *innerME = &(chambME[indexCh]);
// 	      for (int i=-1;i<(phiB->GetNbinsX()+1);i++)
// 		innerME->find(fullName("TrigDirectionPhi"))->second->setBinContent(i,phiB->GetBinContent(i));
// 	      for (int i=-1;i<(phiR->GetNbinsX()+1);i++)
// 		innerME->find(fullName("TrigPositionPhi"))->second->setBinContent(i,phiR->GetBinContent(i));
	     
// 	    }

	    // Perform DCC/DDU common plot analysis (Theta ones)	    
	    TH2F * ThetaBXvsQual = getHisto<TH2F>(dbe->get(getMEName("ThetaBXvsQual","LocalTriggerTheta", chId)));
	    TH1F * ThetaBestQual = getHisto<TH1F>(dbe->get(getMEName("ThetaBestQual","LocalTriggerTheta", chId)));
	
	    // no theta triggers in stat 4!
	    if (ThetaBXvsQual && ThetaBestQual && stat<4) {
	      TH1D* BXH       = ThetaBXvsQual->ProjectionY("",4,4,"");
	      int    BXOK_bin = BXH->GetEffectiveEntries()>=1 ? BXH->GetMaximumBin(): 10;
	      double BX_OK    = ThetaBXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
	      double trigs    = ThetaBestQual->GetEntries(); 
	      double trigsH   = ThetaBestQual->GetBinContent(4);
	      
	      if( secME[sector_id].find(fullName("HFractionSecTheta")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"LocalTriggerTheta","CorrectBXTheta");
		bookSectorHistos(wh,sect,"LocalTriggerTheta","HFractionTheta");
	      }
	      std::map<std::string,MonitorElement*> *innerME = &(secME.find(sector_id)->second);
	      innerME->find(fullName("CorrectBXTheta"))->second->setBinContent(stat,BX_OK);
	      innerME->find(fullName("HFractionTheta"))->second->setBinContent(stat,trigsH/trigs);

	      if( whME[wh].find(fullName("HFractionTheta")) == whME[wh].end() ){
		bookWheelHistos(wh,"LocalTriggerTheta","CorrectBXTheta");
		bookWheelHistos(wh,"LocalTriggerTheta","HFractionTheta");
	      }
	      innerME = &(whME.find(wh)->second);
	      innerME->find(fullName("CorrectBXTheta"))->second->setBinContent(sect,stat,BX_OK);
	      innerME->find(fullName("HFractionTheta"))->second->setBinContent(sect,stat,trigsH/trigs);
	    
	    }

	  }
	}
      }
    }
  }	
  
}


// void DTLocalTriggerTest::bookChambHistos(DTChamberId chambId, string htype) {
  
//   stringstream wheel; wheel << chambId.wheel();
//   stringstream station; station << chambId.station();	
//   stringstream sector; sector << chambId.sector();

//   string fullType  = fullName(htype);
//   string HistoName = fullType + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

//   dbe->setCurrentFolder("DT/Tests/" + testName + "/Wheel" + wheel.str() +
// 			"/Sector" + sector.str() +
// 			"/Station" + station.str());
  
//   uint32_t indexChId = chambId.rawId();
//   if (htype.find("TrigPositionPhi") == 0){
//     chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),100,-500.,500.);
//     return;
//   }
//   if (htype.find("TrigDirectionPhi") == 0){
//     chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),200,-40.,40.);
//     return;
//   }

// }
