/*
 *  See header file for a description of this class.
 *
 *  \author C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
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
  baseFolderTM = "DT/03-LocalTrigger-TM/";
  baseFolderDDU = "DT/04-LocalTrigger-DDU/";
  nMinEvts  = ps.getUntrackedParameter<int>("nEventsCert", 5000);

  bookingdone = 0;

}


DTLocalTriggerTest::~DTLocalTriggerTest(){

}


void DTLocalTriggerTest::Bookings(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  vector<string>::const_iterator iTr   = trigSources.begin();
  vector<string>::const_iterator trEnd = trigSources.end();
  vector<string>::const_iterator iHw   = hwSources.begin();
  vector<string>::const_iterator hwEnd = hwSources.end();

  //Booking
  if(parameters.getUntrackedParameter<bool>("staticBooking", true)){
    for (; iTr != trEnd; ++iTr){
      trigSource = (*iTr);
      for (; iHw != hwEnd; ++iHw){
	hwSource = (*iHw);
	// Loop over the TriggerUnits
	for (int wh=-2; wh<=2; ++wh){
	  if (hwSource=="COM") {

	    bookWheelHistos(ibooker,wh,"MatchingPhi");
	  } 
	  else { 
	    for (int sect=1; sect<=12; ++sect){

	      bookSectorHistos(ibooker,wh,sect,"BXDistribPhi");
	      bookSectorHistos(ibooker,wh,sect,"QualDistribPhi");
	    }

	    bookWheelHistos(ibooker,wh,"CorrectBXPhi");
	    bookWheelHistos(ibooker,wh,"ResidualBXPhi");
	    bookWheelHistos(ibooker,wh,"CorrFractionPhi");
	    bookWheelHistos(ibooker,wh,"2ndFractionPhi");
	    bookWheelHistos(ibooker,wh,"TriggerInclusivePhi");
	  }
	}
      }
    }
  }
  // Summary test histo booking (only static)
  for (iTr = trigSources.begin(); iTr != trEnd; ++iTr){
    trigSource = (*iTr);
    for (iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      for (int wh=-2; wh<=2; ++wh){
	if (hwSource=="COM") {

	  bookWheelHistos(ibooker,wh,"MatchingSummary","Summaries");
	}
	else {

	  bookWheelHistos(ibooker,wh,"CorrFractionSummary","Summaries");
	  bookWheelHistos(ibooker,wh,"2ndFractionSummary","Summaries");
	}
      }
      if (hwSource=="COM") {

	bookCmsHistos(ibooker,"MatchingSummary","Summaries");
      }
      else {

	bookCmsHistos(ibooker,"CorrFractionSummary");
	bookCmsHistos(ibooker,"2ndFractionSummary");
      }
      if (hwSource=="TM") {

	bookCmsHistos(ibooker,"TrigGlbSummary","",true);
	bookCmsHistos(ibooker,"TrigGlbSummary","",true);
      }
       
    }	
  }

  bookingdone = 1; 
}


void DTLocalTriggerTest::beginRun(const edm::Run& r, const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginRun(r,c);

}

void DTLocalTriggerTest::runClientDiagnostic(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  if (!bookingdone) Bookings(ibooker,igetter);

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
	    
	    if (hwSource=="COM") {
	      // Perform TM-DDU matching test and generates summaries (Phi view)
	      TH2F * DDUvsTM = getHisto<TH2F>(igetter.get(getMEName("QualDDUvsQualTM","LocalTriggerPhiIn", chId)));
	      if (DDUvsTM) {
		
		int matchSummary   = 1;
		
		if (DDUvsTM->GetEntries()>1) {
		  
		  double entries     = DDUvsTM->GetEntries();
		  double corrEntries = 0;
		  for (int ibin=2; ibin<=8; ++ibin) {
		    corrEntries += DDUvsTM->GetBinContent(ibin,ibin);
		  }
		  double corrRatio   = corrEntries/entries;
		  
		  if (corrRatio < parameters.getUntrackedParameter<double>("matchingFracError",.65)){
		    matchSummary = 2;
		  }
		  else if (corrRatio < parameters.getUntrackedParameter<double>("matchingFracWarning",.85)){
		    matchSummary = 3;
		  }
		  else {
		    matchSummary = 0;
		  }
		  
		  if( whME[wh].find(fullName("MatchingPhi")) == whME[wh].end() ){
		    bookWheelHistos(ibooker,wh,"MatchingPhi");
		  }
		  
		  whME[wh].find(fullName("MatchingPhi"))->second->setBinContent(sect,stat,corrRatio);
		  
		}
		
		whME[wh].find(fullName("MatchingSummary"))->second->setBinContent(sect,stat,matchSummary);

	      }
	    }
	    else {
	      TH2F * BXvsQual      = getHisto<TH2F>(igetter.get(getMEName("BXvsQual_In","LocalTriggerPhiIn", chId)));
	      TH1F * BestQual      = getHisto<TH1F>(igetter.get(getMEName("BestQual_In","LocalTriggerPhiIn", chId)));
	      TH2F * Flag1stvsQual = getHisto<TH2F>(igetter.get(getMEName("Flag1stvsQual_In","LocalTriggerPhiIn", chId))); 
	      if (BXvsQual && Flag1stvsQual && BestQual) {
		int corrSummary   = 1;
		int secondSummary = 1;
		
		if (BestQual->GetEntries()>1) {
		  TH1D* BXHH    = BXvsQual->ProjectionY("",6,7,"");
		  TH1D* Flag1st = Flag1stvsQual->ProjectionY();
		  int BXOK_bin  = BXHH->GetEntries()>=1 ? BXHH->GetMaximumBin() : 51;
		  double BXMean = BXHH->GetEntries()>=1 ? BXHH->GetMean() : 51;
		  double BX_OK  = BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
		  double trigsFlag2nd = Flag1st->GetBinContent(2);
		  double trigs = Flag1st->GetEntries();
		  double besttrigs = BestQual->GetEntries();
		  double besttrigsCorr = BestQual->Integral(5,7,"");
		  delete BXHH;
		  delete Flag1st;
		  
		  double corrFrac   = besttrigsCorr/besttrigs;
		  double secondFrac = trigsFlag2nd/trigs;
		  if (corrFrac < parameters.getUntrackedParameter<double>("corrFracError",.5)){
		    corrSummary = 2;
		  }
		  else if (corrFrac < parameters.getUntrackedParameter<double>("corrFracWarning",.6)){
		    corrSummary = 3;
		  }
		  else {
		    corrSummary = 0;
		  }
		  if (secondFrac > parameters.getUntrackedParameter<double>("secondFracError",.2)){
		    secondSummary = 2;
		  }
		  else if (secondFrac > parameters.getUntrackedParameter<double>("secondFracWarning",.1)){
		    secondSummary = 3;
		  }
		  else {
		    secondSummary = 0;
		  }
		  
		  if( secME[sector_id].find(fullName("BXDistribPhi")) == secME[sector_id].end() ){
		    bookSectorHistos(ibooker,wh,sect,"QualDistribPhi");
		    bookSectorHistos(ibooker,wh,sect,"BXDistribPhi");
		  }

		  TH1D* BXDistr   = BXvsQual->ProjectionY();
		  TH1D* QualDistr = BXvsQual->ProjectionX();
		  std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
		  
		  int nbinsBX        = BXDistr->GetNbinsX();
		  int firstBinCenter = static_cast<int>(BXDistr->GetBinCenter(1));
		  int lastBinCenter  = static_cast<int>(BXDistr->GetBinCenter(nbinsBX));
		  int iMin = firstBinCenter>-4 ? firstBinCenter : -4;
		  int iMax = lastBinCenter<20  ? lastBinCenter  : 20;
		  for (int ibin=iMin+5;ibin<=iMax+5; ++ibin) {
		    innerME->find(fullName("BXDistribPhi"))->second->setBinContent(ibin,stat,BXDistr->GetBinContent(ibin-5-firstBinCenter+1));
		  }
		  for (int ibin=1;ibin<=7;++ibin) {
		    innerME->find(fullName("QualDistribPhi"))->second->setBinContent(ibin,stat,QualDistr->GetBinContent(ibin));
		  }

		  delete BXDistr;
		  delete QualDistr;

		  if( whME[wh].find(fullName("CorrectBXPhi")) == whME[wh].end() ){
		    bookWheelHistos(ibooker,wh,"ResidualBXPhi");
		    bookWheelHistos(ibooker,wh,"CorrectBXPhi");
		    bookWheelHistos(ibooker,wh,"CorrFractionPhi");
		    bookWheelHistos(ibooker,wh,"2ndFractionPhi");
		    bookWheelHistos(ibooker,wh,"TriggerInclusivePhi");
		  }
		  
		  innerME = &(whME[wh]);
		  innerME->find(fullName("CorrectBXPhi"))->second->setBinContent(sect,stat,BX_OK+0.00001);
		  innerME->find(fullName("ResidualBXPhi"))->second->setBinContent(sect,stat,round(25.*(BXMean-BX_OK))+0.00001);
		  innerME->find(fullName("CorrFractionPhi"))->second->setBinContent(sect,stat,corrFrac);
		  innerME->find(fullName("TriggerInclusivePhi"))->second->setBinContent(sect,stat,besttrigs);
		  innerME->find(fullName("2ndFractionPhi"))->second->setBinContent(sect,stat,secondFrac);
		}

		whME[wh].find(fullName("CorrFractionSummary"))->second->setBinContent(sect,stat,corrSummary);
		whME[wh].find(fullName("2ndFractionSummary"))->second->setBinContent(sect,stat,secondSummary);

	      }

	      if (hwSource=="DDU") {
		TH2F * ThetaBXvsQual = getHisto<TH2F>(igetter.get(getMEName("ThetaBXvsQual","LocalTriggerTheta", chId)));
		TH1F * ThetaBestQual = getHisto<TH1F>(igetter.get(getMEName("ThetaBestQual","LocalTriggerTheta", chId)));
	
		// no theta triggers in stat 4!
		if (ThetaBXvsQual && ThetaBestQual && stat<4 && ThetaBestQual->GetEntries()>1) {
		  TH1D* BXH       = ThetaBXvsQual->ProjectionY("",4,4,"");
		  int    BXOK_bin = BXH->GetEffectiveEntries()>=1 ? BXH->GetMaximumBin(): 10;
		  double BX_OK    = ThetaBXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
		  double trigs    = ThetaBestQual->GetEntries(); 
		  double trigsH   = ThetaBestQual->GetBinContent(4);
		  delete BXH; 
				
		  if( whME[wh].find(fullName("HFractionTheta")) == whME[wh].end() ){
		    bookWheelHistos(ibooker,wh,"CorrectBXTheta");
		    bookWheelHistos(ibooker,wh,"HFractionTheta");
		  }
		  std::map<std::string,MonitorElement*> *innerME = &(whME.find(wh)->second);
		  innerME->find(fullName("CorrectBXTheta"))->second->setBinContent(sect,stat,BX_OK+0.00001);
		  innerME->find(fullName("HFractionTheta"))->second->setBinContent(sect,stat,trigsH/trigs);
		
		}
	      }
	      else if (hwSource=="TM") {
		// Perform TM plot analysis (Theta ones)	    
		TH2F * ThetaPosvsBX = getHisto<TH2F>(igetter.get(getMEName("PositionvsBX","LocalTriggerTheta", chId)));
	      
		// no theta triggers in stat 4!
		if (ThetaPosvsBX && stat<4 && ThetaPosvsBX->GetEntries()>1) {
		  TH1D* BX        = ThetaPosvsBX->ProjectionX();
		  int    BXOK_bin = BX->GetEffectiveEntries()>=1 ? BX->GetMaximumBin(): 10;
		  double BX_OK    = ThetaPosvsBX->GetXaxis()->GetBinCenter(BXOK_bin);
		  delete BX; 
		
		  if( whME[wh].find(fullName("CorrectBXTheta")) == whME[wh].end() ){
		    bookWheelHistos(ibooker,wh,"CorrectBXTheta");
		  }
		  std::map<std::string,MonitorElement*> *innerME = &(whME.find(wh)->second);
		  innerME->find(fullName("CorrectBXTheta"))->second->setBinContent(sect,stat,BX_OK+0.00001);
		}
            // After TM the DDU is not used and the TM has information on the Theta Quality
            // Adding trigger info to compute H fraction (11/10/2016) M.C.Fouz
		TH2F * ThetaBXvsQual = getHisto<TH2F>(igetter.get(getMEName("ThetaBXvsQual","LocalTriggerTheta", chId)));
		TH1F * ThetaBestQual = getHisto<TH1F>(igetter.get(getMEName("ThetaBestQual","LocalTriggerTheta", chId)));
		if (ThetaBXvsQual && ThetaBestQual && stat<4 && ThetaBestQual->GetEntries()>1) {
		  double trigs    = ThetaBestQual->GetEntries(); 
		  double trigsH   = ThetaBestQual->GetBinContent(2); // Note that for the new plots H is at bin=2 and not 4 as in DDU!!!!
		  if( whME[wh].find(fullName("HFractionTheta")) == whME[wh].end() ){
		      bookWheelHistos(ibooker,wh,"HFractionTheta");
		  }
		  std::map<std::string,MonitorElement*> *innerME = &(whME.find(wh)->second);
		  innerME->find(fullName("HFractionTheta"))->second->setBinContent(sect,stat,trigsH/trigs);
	      }
            // END ADDING H Fraction info
            }
          }

        }
      }
      }
    }
  }


  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);  
      for (int wh=-2; wh<=2; ++wh){
	std::map<std::string,MonitorElement*> *innerME = &(whME[wh]);
	if(hwSource=="COM") {
	  TH2F* matchWhSummary   = getHisto<TH2F>(innerME->find(fullName("MatchingSummary"))->second);
	  for (int sect=1; sect<=12; ++sect){
	    int matchErr      = 0;
	    int matchNoData   = 0;
	    for (int stat=1; stat<=4; ++stat){
	      switch (static_cast<int>(matchWhSummary->GetBinContent(sect,stat))) {
	      case 1:
		matchNoData++;
	      case 2:
		matchErr++;
	      }
	    }
	    if (matchNoData == 4)   matchErr   = 5;
	    cmsME.find(fullName("MatchingSummary"))->second->setBinContent(sect,wh+3,matchErr);
	  }
	}
	else {
	  TH2F* corrWhSummary   = getHisto<TH2F>(innerME->find(fullName("CorrFractionSummary"))->second);
	  TH2F* secondWhSummary = getHisto<TH2F>(innerME->find(fullName("2ndFractionSummary"))->second);
	  for (int sect=1; sect<=12; ++sect){
	    int corrErr      = 0;
	    int secondErr    = 0;
	    int corrNoData   = 0;
	    int secondNoData = 0;
	    for (int stat=1; stat<=4; ++stat){
	      switch (static_cast<int>(corrWhSummary->GetBinContent(sect,stat))) {
	      case 1:
		corrNoData++;
	      case 2:
		corrErr++;
	      }
	      switch (static_cast<int>(secondWhSummary->GetBinContent(sect,stat))) {
	      case 1:
		secondNoData++;
	      case 2:
		secondErr++;
	      }
	    }
	    if (corrNoData == 4)   corrErr   = 5;
	    if (secondNoData == 4) secondErr = 5;
	    cmsME.find(fullName("CorrFractionSummary"))->second->setBinContent(sect,wh+3,corrErr);
	    cmsME.find(fullName("2ndFractionSummary"))->second->setBinContent(sect,wh+3,secondErr);
	  }
	}
      }
    }
  }

  fillGlobalSummary(igetter);

}

void DTLocalTriggerTest::fillGlobalSummary(DQMStore::IGetter & igetter) {

  float glbPerc[5] = { 1., 0.9, 0.6, 0.3, 0.01 };
  trigSource = "";
  hwSource = "TM";  

  int nSecReadout = 0;

  for (int wh=-2; wh<=2; ++wh) {
    for (int sect=1; sect<=12; ++sect) {

      float maxErr = 8.;
      int corr   = cmsME.find(fullName("CorrFractionSummary"))->second->getBinContent(sect,wh+3);
      int second = cmsME.find(fullName("2ndFractionSummary"))->second->getBinContent(sect,wh+3);
      int lut=0;
      MonitorElement * lutsME = igetter.get(topFolder(hwSource=="TM") + "Summaries/TrigLutSummary");
      if (lutsME) {
	lut = lutsME->getBinContent(sect,wh+3);
	maxErr+=4;
      } else {
	LogTrace(category()) << "[" << testName 
	 << "Test]: TM Lut test Summary histo not found." << endl;
      }
      (corr <5 || second<5) && nSecReadout++;
      int errcode = ((corr<5 ? corr : 4) + (second<5 ? second : 4) + (lut<5 ? lut : 4) );
      errcode = min(int((errcode/maxErr + 0.01)*5),5);
      cmsME.find("TrigGlbSummary")->second->setBinContent(sect,wh+3,glbPerc[errcode]);
    
    }
  }

  if (!nSecReadout) 
    cmsME.find("TrigGlbSummary")->second->Reset(); // white histo id TM is not RO
  
  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsTrigger";
  MonitorElement * meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    cmsME.find("TrigGlbSummary")->second->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    cmsME.find("TrigGlbSummary")->second->setEntries(nMinEvts + 1);
    LogVerbatim (category()) << "[" << testName 
	 << "Test]: ME: " <<  nEvtsName << " not found!" << endl;
  }

}



