/*
 *  See header file for a description of this class.
 *
 *  \author D. Fasanella - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */


// This class header
#include "DQM/DTMonitorClient/src/DTTriggerLutTest.h"

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
//#include "TSpectrum.h"


//C++ headers
#include <iostream>
#include <sstream>


using namespace edm;
using namespace std;


DTTriggerLutTest::DTTriggerLutTest(const edm::ParameterSet& ps){
	
  setConfig(ps,"DTTriggerLut");
  baseFolderDCC = "DT/03-LocalTrigger-DCC/";
  baseFolderDDU = "DT/04-LocalTrigger-DDU/";
  thresholdWarnPhi  = ps.getUntrackedParameter<double>("thresholdWarnPhi");
  thresholdErrPhi   = ps.getUntrackedParameter<double>("thresholdErrPhi");
  thresholdWarnPhiB = ps.getUntrackedParameter<double>("thresholdWarnPhiB");
  thresholdErrPhiB  = ps.getUntrackedParameter<double>("thresholdErrPhiB");
  validRange = ps.getUntrackedParameter<double>("validRange");
  detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis");

  bookingdone = 0;

}


DTTriggerLutTest::~DTTriggerLutTest(){

}


void DTTriggerLutTest::Bookings(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  bookingdone = 1;  

  vector<string>::const_iterator iTr   = trigSources.begin();
  vector<string>::const_iterator trEnd = trigSources.end();
  vector<string>::const_iterator iHw   = hwSources.begin();
  vector<string>::const_iterator hwEnd = hwSources.end();
  
  //Booking
  if(parameters.getUntrackedParameter<bool>("staticBooking")){
    
    for (; iTr != trEnd; ++iTr){
      trigSource = (*iTr);
      for (; iHw != hwEnd; ++iHw){
	hwSource = (*iHw);
	// Loop over the TriggerUnits
	for (int wh=-2; wh<=2; ++wh){
	  if (detailedAnalysis){
	    bookWheelHistos(ibooker,wh,"PhiResidualPercentage");  
	    bookWheelHistos(ibooker,wh,"PhibResidualPercentage"); 
	  }
   
	  bookWheelHistos(ibooker,wh,"PhiLutSummary","Summaries");
	  bookWheelHistos(ibooker,wh,"PhibLutSummary","Summaries");      
	  
	  if (detailedAnalysis){
	    bookWheelHistos(ibooker,wh,"PhiResidualMean");  
	    bookWheelHistos(ibooker,wh,"PhiResidualRMS");
	    bookWheelHistos(ibooker,wh,"PhibResidualMean");  
	    bookWheelHistos(ibooker,wh,"PhibResidualRMS");
	    bookWheelHistos(ibooker,wh,"CorrelationFactorPhi");
	    bookWheelHistos(ibooker,wh,"CorrelationFactorPhib");
	    bookWheelHistos(ibooker,wh,"DoublePeakFlagPhib");
	  }

	}

	bookCmsHistos(ibooker,"TrigLutSummary","",true);
	bookCmsHistos(ibooker,"PhiLutSummary");
	bookCmsHistos(ibooker,"PhibLutSummary");
	if (detailedAnalysis){

	  bookCmsHistos1d(ibooker,"PhiPercentageSummary");
	  bookCmsHistos1d(ibooker,"PhibPercentageSummary");
	}
      }
    }
  }	
}


void DTTriggerLutTest::beginRun(const edm::Run& r, const edm::EventSetup& c){

  DTLocalTriggerBaseTest::beginRun(r,c);
  
}

void DTTriggerLutTest::runClientDiagnostic(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

   if (!bookingdone) Bookings(ibooker,igetter);

  // Reset lut percentage 1D summaries
  if (detailedAnalysis){
    cmsME.find(fullName("PhiPercentageSummary"))->second->Reset();
    cmsME.find(fullName("PhibPercentageSummary"))->second->Reset();
  }

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      vector<const DTChamber*>::const_iterator chIt  = muonGeom->chambers().begin();
      vector<const DTChamber*>::const_iterator chEnd = muonGeom->chambers().end();
      for (; chIt != chEnd; ++chIt) {

	DTChamberId chId((*chIt)->id());
	int wh   = chId.wheel();
	int sect = chId.sector();
	int stat = chId.station();
	
	std::map<std::string,MonitorElement*> &innerME = whME[wh];
	  
	// Make Phi Residual Summary
	TH1F * PhiResidual = getHisto<TH1F>(igetter.get(getMEName("PhiResidual","Segment", chId)));
	int phiSummary = 1;
	if (PhiResidual && PhiResidual->GetEntries()>10) {

	  if( innerME.find(fullName("PhiResidualPercentage")) == innerME.end() ){
	    bookWheelHistos(ibooker,wh,"PhiResidualPercentage");  
	  }
	  
	  float rangeBin = validRange/(PhiResidual->GetBinWidth(1));
	  float center   = (PhiResidual->GetNbinsX())/2.;
	  float perc     = (PhiResidual->Integral(floor(center-rangeBin),ceil(center+rangeBin)))/(PhiResidual->Integral());
	  fillWhPlot(innerME.find(fullName("PhiResidualPercentage"))->second,sect,stat,perc,false);
	  phiSummary = performLutTest(perc,thresholdWarnPhi,thresholdErrPhi);
	  if (detailedAnalysis) cmsME.find(fullName("PhiPercentageSummary"))->second->Fill(perc);

	}

	fillWhPlot(innerME.find(fullName("PhiLutSummary"))->second,sect,stat,phiSummary);
	
	if (detailedAnalysis){

	  if ((phiSummary==0)||(phiSummary==3)){ //Information on the Peak

	    if( innerME.find(fullName("PhiResidualMean")) == innerME.end() ){ 
	      bookWheelHistos(ibooker,wh,"PhiResidualMean");  
	      bookWheelHistos(ibooker,wh,"PhiResidualRMS");  
	    }

	    float center   = (PhiResidual->GetNbinsX())/2.;                   
	    float rangeBin = validRange/(PhiResidual->GetBinWidth(1));
	    PhiResidual->GetXaxis()->SetRange(floor(center-rangeBin),ceil(center+rangeBin));
	    float max     = PhiResidual->GetMaximumBin();
	    float maxBin  = PhiResidual->GetXaxis()->FindBin(max);
	    float nBinMax = 0.5/(PhiResidual->GetBinWidth(1));
	    PhiResidual->GetXaxis()->SetRange(floor(maxBin-nBinMax),ceil(maxBin+nBinMax));
	    float Mean = PhiResidual->GetMean();
	    float rms  = PhiResidual->GetRMS();	    

	    fillWhPlot(innerME.find(fullName("PhiResidualMean"))->second,sect,stat,Mean);
	    fillWhPlot(innerME.find(fullName("PhiResidualRMS"))->second,sect,stat,rms);
	    
	  }
	  
	  TH2F * TrackPhitkvsPhitrig   = getHisto<TH2F>(igetter.get(getMEName("PhitkvsPhitrig","Segment", chId)));

	  if (TrackPhitkvsPhitrig && TrackPhitkvsPhitrig->GetEntries()>100) {
	    float corr = TrackPhitkvsPhitrig->GetCorrelationFactor();
	    if( innerME.find(fullName("CorrelationFactorPhi")) == innerME.end() ){
	      bookWheelHistos(ibooker,wh,"CorrelationFactorPhi");
	    }
	    fillWhPlot(innerME.find(fullName("CorrelationFactorPhi"))->second,sect,stat,corr,false);
	  }
	  
	}
	
				
	// Make Phib Residual Summary
	TH1F * PhibResidual = getHisto<TH1F>(igetter.get(getMEName("PhibResidual","Segment", chId)));
	int phibSummary = stat==3 ? -1 : 1; // station 3 has no meaningful MB3 phi bending information
	
	if (stat != 3 && PhibResidual && PhibResidual->GetEntries()>10) {// station 3 has no meaningful MB3 phi bending information

	  if( innerME.find(fullName("PhibResidualPercentage")) == innerME.end() ){ 
	    bookWheelHistos(ibooker,wh,"PhibResidualPercentage");  
	  }
	  
	  float rangeBin = validRange/(PhibResidual->GetBinWidth(1));
	  float center   = (PhibResidual->GetNbinsX())/2.;
	  float perc     = (PhibResidual->Integral(floor(center-rangeBin),ceil(center+rangeBin)))/(PhibResidual->Integral());

	  fillWhPlot(innerME.find(fullName("PhibResidualPercentage"))->second,sect,stat,perc,false);
	  phibSummary = performLutTest(perc,thresholdWarnPhiB,thresholdErrPhiB);
	  if (detailedAnalysis) cmsME.find(fullName("PhibPercentageSummary"))->second->Fill(perc);

	}

	fillWhPlot(innerME.find(fullName("PhibLutSummary"))->second,sect,stat,phibSummary);
	
	if (detailedAnalysis){	  
	  
	  if ((phibSummary==0)||(phibSummary==3)){

	    if( innerME.find(fullName("PhibResidualMean")) == innerME.end() ){  
	      bookWheelHistos(ibooker,wh,"PhibResidualMean");  
	      bookWheelHistos(ibooker,wh,"PhibResidualRMS");  
	    }

	    float center   = (PhibResidual->GetNbinsX())/2.;
	    float rangeBin =  validRange/(PhibResidual->GetBinWidth(1));
	    PhibResidual->GetXaxis()->SetRange(floor(center-rangeBin),ceil(center+rangeBin));
	    float max     = PhibResidual->GetMaximumBin();
	    float maxBin  = PhibResidual->GetXaxis()->FindBin(max);
	    float nBinMax = 0.5/(PhibResidual->GetBinWidth(1));
	    PhibResidual->GetXaxis()->SetRange(floor(maxBin-nBinMax),ceil(maxBin+nBinMax));
	    float Mean = PhibResidual->GetMean();
	    float rms = PhibResidual->GetRMS();
	    
	    fillWhPlot(innerME.find(fullName("PhibResidualMean"))->second,sect,stat,Mean);
	    fillWhPlot(innerME.find(fullName("PhibResidualRMS"))->second,sect,stat,rms);
	  }

	  TH2F * TrackPhibtkvsPhibtrig   = getHisto<TH2F>(igetter.get(getMEName("PhibtkvsPhibtrig","Segment", chId)));
	  if (TrackPhibtkvsPhibtrig && TrackPhibtkvsPhibtrig->GetEntries()>100) {

	    float corr = TrackPhibtkvsPhibtrig->GetCorrelationFactor();
	    if( innerME.find(fullName("CorrelationFactorPhib")) == innerME.end() ){
	      bookWheelHistos(ibooker,wh,"CorrelationFactorPhib");
	    }

	    fillWhPlot(innerME.find(fullName("CorrelationFactorPhib"))->second,sect,stat,corr,false);

	  }	  
	  
	}						
      }
    }
  }
	
  // Barrel Summary Plots
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);  
      for (int wh=-2; wh<=2; ++wh){

	std::map<std::string,MonitorElement*> *innerME = &(whME[wh]);
	
	TH2F* phiWhSummary   = getHisto<TH2F>(innerME->find(fullName("PhiLutSummary"))->second);
	TH2F* phibWhSummary  = getHisto<TH2F>(innerME->find(fullName("PhibLutSummary"))->second);

	for (int sect=1; sect<=12; ++sect){

	  int phiSectorTotal  = 0;   // CB dai 1 occhio a questo
	  int phibSectorTotal = 0;
	  int nullphi  = 0;
	  int nullphib = 0;
	  int phiStatus  = 5;
	  int phibStatus = 5;
	  int glbStatus  = 0;

	  for (int stat=1; stat<=4; ++stat){
	    if (phiWhSummary->GetBinContent(sect,stat)==2){
	      phiSectorTotal +=1;
	      glbStatus += 1;
	    }
	    if (phiWhSummary->GetBinContent(sect,stat)==1)
	      nullphi+=1;
	    if (phibWhSummary->GetBinContent(sect,stat)==2) {
	      phibSectorTotal+=1;
	      glbStatus += 1;
	    }
	    if (phibWhSummary->GetBinContent(sect,stat)==1)
	      nullphib+=1;
	  }
	  if (nullphi!=4)
	    phiStatus=phiSectorTotal;
	  else 
	    phiStatus=5;
	  if (nullphib!=3)
	    phibStatus=phibSectorTotal;
	  else 
	    phibStatus=5;
	  
	  cmsME.find("TrigLutSummary")->second->setBinContent(sect,wh+3,glbStatus);
	  cmsME.find(fullName("PhiLutSummary"))->second->setBinContent(sect,wh+3,phiStatus);
	  cmsME.find(fullName("PhibLutSummary"))->second->setBinContent(sect,wh+3,phibStatus);
	}
      }
    }
  }

}

int DTTriggerLutTest::performLutTest(double perc,double thresholdWarn ,double thresholdErr) {
	
  bool isInWarn = perc<thresholdWarn;  
  bool isInErr  = perc<thresholdErr;

  int result= isInErr ? 2 : isInWarn ? 3 : 0;

  return result;

}

void DTTriggerLutTest::bookCmsHistos1d(DQMStore::IBooker & ibooker, string hTag, string folder) {

  string basedir = topFolder(true);
  if (folder != "") {
    basedir += folder +"/" ;
  }
  ibooker.setCurrentFolder(basedir);

  string hName = fullName(hTag);
  LogTrace(category()) << "[" << testName << "Test]: booking " << basedir << hName;

  MonitorElement* me = ibooker.book1D(hName.c_str(),hName.c_str(),101,-0.005,1.005);
  me->setAxisTitle("Percentage",1);
  cmsME[hName] = me;

}

void DTTriggerLutTest::fillWhPlot(MonitorElement *plot, int sect, int stat, float value, bool lessIsBest) {
  
  if (sect>12) {
    int scsect = sect==13 ? 4 : 10;
    if ((value>plot->getBinContent(scsect,stat)) == lessIsBest) {
      plot->setBinContent(scsect,stat,value);
    }
  }
  else {
    plot->setBinContent(sect,stat,value);
  }
  
  return;
  
}


