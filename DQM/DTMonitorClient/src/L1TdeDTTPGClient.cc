/*
 * \file L1TdeDTTPGClient.cc
 * 
  * $Date: 2010/11/18 09:42:52 $
 * $Revision: 1.0 $
 * \author C. Battilana - CIEMAT
 * \author M. Meneghelli - INFN BO
 *
*/

#include "DQM/DTMonitorClient/src/L1TdeDTTPGClient.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

//Root
#include"TH1.h"
#include"TH1F.h"
#include"TAxis.h"

//C++
#include <sstream>
#include <iostream>
#include <math.h>

using namespace edm;
using namespace std;

L1TdeDTTPGClient::L1TdeDTTPGClient(const ParameterSet& parameters) {
  
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: Constructor"<<endl;

  theBaseFolder  = "L1TEMU/DTTPGexpert";
  theParams      = parameters;
  theDQMStore    = Service<DQMStore>().operator->();
  
  theRunOnline  = parameters.getUntrackedParameter<bool>("runOnline");
  theQualTh     = parameters.getUntrackedParameter<double>("qualThreshold");
  theStatQualTh = parameters.getUntrackedParameter<double>("statQualThreshold");
  thePhiTh      = parameters.getUntrackedParameter<double>("phiThreshold");
  thePhiBendTh  = parameters.getUntrackedParameter<double>("phibendThreshold");
  

}


L1TdeDTTPGClient::~L1TdeDTTPGClient() {

  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: Destructor"<< endl;

}


void L1TdeDTTPGClient::beginJob(){
 
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: BeginJob" << endl;
  theLumis  = 0;

}


void L1TdeDTTPGClient::beginRun(const Run& run, const EventSetup& context) {

  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: BeginRun" << endl;   
  for (int wh=-2;wh<=2;++wh){ 
    bookWheelHistos(wh); 
  }
  bookBarrelHistos();
		
}


void L1TdeDTTPGClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: Begin of LS transition" << endl;
  theLumis++;
  
}

void L1TdeDTTPGClient::endLuminosityBlock(const LuminosityBlock&  lumiSeg, const  EventSetup& context){

  if (theRunOnline)
    performClientDiagnostic();

}

void L1TdeDTTPGClient::performClientDiagnostic(){

  barrelHistos["hFracHasBoth"]->Reset();
  barrelHistos["hFracQualInRange"]->Reset();
  barrelHistos["hQualStatAgreement"]->Reset();
  barrelHistos["hFracPhiInRange"]->Reset();
  barrelHistos["hFracPhiBendInRange"]->Reset();


  
  for (int wh=-2;wh<=2;++wh){
    for (int sec=1;sec<=12;++sec){
      for (int st=1;st<=4;++st){
	DTChamberId chId(wh,st,sec);
	
	float qualInRange    = fracInRange(getHisto(chId,"hDeltaQuality"),1);
	float phiInRange     = fracInRange(getHisto(chId,"hDeltaPhi"),2);
	float phiBendInRange = st != 3 ? fracInRange(getHisto(chId,"hDeltaPhiBend"),2) : 1.;

	TH1F* hEntries = getHisto(chId,"hEntries")->getTH1F();
	float hasBothFrac = hEntries->Integral()>0 ? 
	  hEntries->GetBinContent(1)/hEntries->Integral() : 0.;

	float statAgr = computeAgreement(getHisto(chId,"hDataQuality"),
					 getHisto(chId,"hEmuQuality"));
	
	float summary = qualInRange>theQualTh && hasBothFrac>theHasBothTh && statAgr>theStatQualTh 
	                && phiInRange >= thePhiTh && phiBendInRange >= thePhiBendTh ? 1. : 0. ; 
	
	int phiBendId = st==4 ? 3 : st;  
	whHistos[wh]["hQualStatAgreementWheelMap"]->setBinContent(sec,st,statAgr);
	whHistos[wh]["hDeltaQualityWheelMap"]->setBinContent(sec,st,qualInRange);
	whHistos[wh]["hEntriesWheelMap"]->setBinContent(sec,st,hasBothFrac);
	whHistos[wh]["hTestSummary"]->setBinContent(sec,st,summary);
	whHistos[wh]["hDeltaPhiWheelMap"]->setBinContent(sec,st,phiInRange);
	if (st!=3) {
	  whHistos[wh]["hDeltaPhiBendWheelMap"]->setBinContent(sec,phiBendId,phiBendInRange);
	}
	barrelHistos["hFracHasBoth"]->Fill(hasBothFrac);
	barrelHistos["hFracQualInRange"]->Fill(qualInRange);
	barrelHistos["hQualStatAgreement"]->Fill(statAgr);
	barrelHistos["hFracPhiInRange"]->Fill(phiInRange);
	barrelHistos["hFracPhiBendInRange"]->Fill(phiBendInRange);
      }
    }	
  }  

}


void L1TdeDTTPGClient::endRun(const Run& run, const EventSetup& context) {

  if(!theRunOnline)
    performClientDiagnostic();

}


void L1TdeDTTPGClient::endJob(){
  
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: analyzed " << theLumis << " lumi sections" << endl;
  
}


void L1TdeDTTPGClient::analyze(const Event& event, const EventSetup& context){  
  
}


void L1TdeDTTPGClient::bookWheelHistos(int wh) {

  stringstream wheel; wheel << wh;	

  string path = topFolder() + "/Wheel" + wheel.str();
  
  theDQMStore->setCurrentFolder(path);
  
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: booking histos in : " 
			 << path << endl;

  string chTag = "_W" + wheel.str() ;
  
  whHistos[wh]["hDeltaQualityWheelMap"] = 
    theDQMStore->book2D("hDeltaQualityWheelMap"+chTag,"Data - Emu quality difference summary ",12,1,13,4,1,5);
  whHistos[wh]["hEntriesWheelMap"]      = 
    theDQMStore->book2D("hEntriesWheelMap"+chTag,"Occupancy summary",12,1,13,4,1,5);
  whHistos[wh]["hQualStatAgreementWheelMap"] = 
    theDQMStore->book2D("hQualStatAgreementWheelMap"+chTag,"Quality Agreement difference summary ",12,1,13,4,1,5);
  whHistos[wh]["hTestSummary"]          = 
    theDQMStore->book2D("hTestSummary"+chTag,"Occupancy summary",12,1,13,4,1,5);
  whHistos[wh]["hDeltaPhiWheelMap"]     = 
    theDQMStore->book2D("hDeltaPhiWheelMap"+chTag,"Data - Emu phi assignement difference summary ",12,1,13,4,1,5);
  whHistos[wh]["hDeltaPhiBendWheelMap"] = 
    theDQMStore->book2D("hDeltaPhiBendWheelMap"+chTag,"Data - Emu phi bending assignement  difference summary",12,1,13,3,1,4);
  
}


MonitorElement * L1TdeDTTPGClient::getHisto(const DTChamberId & chId, string  histoTag) const {

  stringstream wheel; wheel << chId.wheel();	
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();	

  string folderName =  theBaseFolder + "/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";
  
  string histoName = folderName + histoTag +
    + "_W" + wheel.str() 
    + "_Sec" + sector.str()
    + "_St" + station.str();

  MonitorElement *histo = theDQMStore->get(histoName);
    
  return histo;

}


void L1TdeDTTPGClient::bookBarrelHistos() {

  theDQMStore->setCurrentFolder(topFolder());
  
  LogTrace("L1TdeDTTPGClient") << "[L1TdeDTTPGClient]: booking histos in : " 
			 << topFolder() << endl;  

  barrelHistos["hFracHasBoth"] = 
    theDQMStore->book1D("hFracHasBoth","Fraction of events with both data & emu",101,-.005,1.005);
  barrelHistos["hFracQualInRange"] = 
    theDQMStore->book1D("hFracQualInRange","Fraction of events where quality matching is OK",101,-.005,1.005);
  barrelHistos["hFracPhiInRange"] = 
    theDQMStore->book1D("hFracPhiInRange","Fraction of events where phi matching is OK",101,-.005,1.005);
  barrelHistos["hFracPhiBendInRange"] = 
    theDQMStore->book1D("hFracPhiBendInRange","Fraction of events where phi bending matching OK",101,-.005,1.005);
  barrelHistos["hQualStatAgreement"] = 
    theDQMStore->book1D("hQualStatAgreement","Quality Statistical Agreement Distrib",101,-.005,1.005);

}


float L1TdeDTTPGClient::fracInRange(MonitorElement *me, int range) {

  TH1F *histo = me->getTH1F();

  float entries = histo->Integral();
  float entriesInRange = histo->Integral(histo->FindBin(-range),histo->FindBin(range));
  float fracInRange =  entries>20 ?  entriesInRange/entries : -1. ;
  
  return fracInRange;

}


float L1TdeDTTPGClient::computeAgreement(MonitorElement *data , MonitorElement *emu) {

  if ( data->getNbinsX()!=emu->getNbinsX() ) {
    LogPrint("L1TdeDTTPGClient") << 
      "[L1TdeDTTPGClient]: data & emu plots have different # of bins!" << endl;
    return -1.;
  }

  int nBins = data->getNbinsX();

  double delta = 0.;
  for (int iBin=1; iBin<=nBins; ++iBin) {
    delta += fabs(data->getBinContent(iBin) - emu->getBinContent(iBin));
  }
  delta /= (data->getEntries() + emu->getEntries());
  float matching = data->getEntries()>20 || emu->getEntries()>20 ? max(1-delta,0.) : -1;
  
  return matching;	   

}
