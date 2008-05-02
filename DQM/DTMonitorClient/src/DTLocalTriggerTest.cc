
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

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: Constructor";

  sourceFolder = ps.getUntrackedParameter<string>("folderRoot", ""); 
  hwSource = ps.getUntrackedParameter<bool>("dataFromDDU", false) ? "DDU" : "DCC" ; 
  parameters = ps;
  dbe = edm::Service<DQMStore>().operator->();
  dbe->setVerbose(1);

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}


DTLocalTriggerTest::~DTLocalTriggerTest(){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: analyzed " << nevents << " events";

}


void DTLocalTriggerTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: BeginJob";
  nevents = 0;
  context.get<MuonGeometryRecord>().get(muonGeom);
  
}


void DTLocalTriggerTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("localTrigger") <<"[DTLocalTriggerTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTLocalTriggerTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: "<<nevents<<" events";

}


void DTLocalTriggerTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //   nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;


  edm::LogVerbatim ("localTrigger") <<"[DTLocalTriggerTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("localTrigger") <<"[DTLocalTriggerTest]: "<<nLumiSegs<<" updates";
  

  // Loop over the TriggerUnits
  for (int stat=1; stat<=4; ++stat){
    for (int wh=-2; wh<=2; ++wh){
      for (int sect=1; sect<=12; ++sect){
	DTChamberId chId(wh,stat,sect);
	int sector_id = (wh+3)+(sect-1)*5;
	uint32_t indexCh = chId.rawId();

	// Perform DCC/DDU common plot analysis (Phi ones)
	TH2F * BXvsQual    = getHisto<TH2F>(dbe->get(getMEName("BXvsQual1st","LocalTriggerPhi", chId)));
	TH1F * BestQual    = getHisto<TH1F>(dbe->get(getMEName("BestQual","LocalTriggerPhi", chId)));
	TH2F * Flag1stvsBX = getHisto<TH2F>(dbe->get(getMEName("Flag1stvsBX","LocalTriggerPhi", chId)));
	if (BXvsQual && Flag1stvsBX && BestQual) {
	      
	  TH1D* BXHH    = BXvsQual->ProjectionY("",7,7,"");
	  TH1D* Flag1st = Flag1stvsBX->ProjectionY();
	  int BXOK_bin = BXHH->GetEntries()>=1 ? BXHH->GetMaximumBin() : 51;
	  double BX_OK =  BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
	  double trigsFlag2nd = Flag1st->GetBinContent(2);
	  double trigs = Flag1st->GetEntries();
	  double besttrigs = BestQual->GetEntries();
	  double besttrigsCorr = 0;
	  for (int i=5;i<=7;++i)
	    besttrigsCorr+=BestQual->GetBinContent(i);
	      
	  if( secME[sector_id].find("CorrectBXSec_Phi") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrectBXSec_Phi");
	    bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrFractionSec_Phi");
	    bookSectorHistos(wh,sect,"LocalTriggerPhi","2ndFractionSec_Phi");
	  }
	  if( whME[wh].find("CorrectBX_Phi") == whME[wh].end() ){
	    bookWheelHistos(wh,"LocalTriggerPhi","CorrectBX_Phi");
	    bookWheelHistos(wh,"LocalTriggerPhi","CorrFraction_Phi");
	    bookWheelHistos(wh,"LocalTriggerPhi","2ndFraction_Phi");
	  }
	  std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	  innerME.find("CorrectBXSec_Phi")->second->setBinContent(stat,BX_OK);
	  innerME.find("CorrFractionSec_Phi")->second->setBinContent(stat,besttrigsCorr/besttrigs);
	  innerME.find("2ndFractionSec_Phi")->second->setBinContent(stat,trigsFlag2nd/trigs);
	   
	  innerME = whME[wh];
	  int pos = stat+4*(sect-1);
	  innerME.find("CorrectBX_Phi")->second->setBinContent(pos,BX_OK);
	  innerME.find("CorrFraction_Phi")->second->setBinContent(pos,besttrigsCorr/besttrigs);
	  innerME.find("2ndFraction_Phi")->second->setBinContent(pos,trigsFlag2nd/trigs);
	    
	}

	// Perform analysis on DCC exclusive plots (Phi)	  
	TH2F * QualvsPhirad  = getHisto<TH2F>(dbe->get(getMEName("QualvsPhirad","LocalTriggerPhi", chId)));
	TH2F * QualvsPhibend = getHisto<TH2F>(dbe->get(getMEName("QualvsPhibend","LocalTriggerPhi", chId)));
	if (QualvsPhirad && QualvsPhibend) {
	      
	  TH1D* phiR = QualvsPhirad->ProjectionX();
	  TH1D* phiB = QualvsPhibend->ProjectionX("_px",5,7,"");

	  if( chambME[indexCh].find("TrigDirection_Phi") == chambME[indexCh].end() ){
	    bookChambHistos(chId,"TrigDirection_Phi");
	    bookChambHistos(chId,"TrigPosition_Phi");
	  }
	  std::map<std::string,MonitorElement*> innerME = chambME[indexCh];
	  for (int i=-1;i<(phiB->GetNbinsX()+1);i++)
	    innerME.find("TrigDirection_Phi")->second->setBinContent(i,phiB->GetBinContent(i));
	  for (int i=-1;i<(phiR->GetNbinsX()+1);i++)
	    innerME.find("TrigPosition_Phi")->second->setBinContent(i,phiR->GetBinContent(i));
	     
	}

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
	      
	  if( secME[sector_id].find("HFractionSec_Theta") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"LocalTriggerTheta","CorrectBXSec_Theta");
	    bookSectorHistos(wh,sect,"LocalTriggerTheta","HFractionSec_Theta");
	  }
	  std::map<std::string,MonitorElement*> innerME = secME.find(sector_id)->second;
	  innerME.find("CorrectBXSec_Theta")->second->setBinContent(stat,BX_OK);
	  innerME.find("HFractionSec_Theta")->second->setBinContent(stat,trigsH/trigs);

	  if( whME[wh].find("HFraction_Theta") == whME[wh].end() ){
	    bookWheelHistos(wh,"LocalTriggerTheta","CorrectBX_Theta");
	    bookWheelHistos(wh,"LocalTriggerTheta","HFraction_Theta");
	  }
	  int pos = stat+3*(sect-1);
	  innerME = whME.find(wh)->second;
	  innerME.find("CorrectBX_Theta")->second->setBinContent(pos,BX_OK);
	  innerME.find("HFraction_Theta")->second->setBinContent(pos,trigsH/trigs);
	    
	}

	// Perform Efficiency analysis (Phi+Segments 2D)
	TH2F * TrackPosvsAngle            = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngle","Segment", chId)));
	TH2F * TrackPosvsAngleandTrig     = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleandTrig","Segment", chId)));
	TH2F * TrackPosvsAngleandTrigHHHL = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleandTrigHHHL","Segment", chId)));
	    
	if (TrackPosvsAngle && TrackPosvsAngleandTrig && TrackPosvsAngleandTrigHHHL) {
	      
	  if( chambME[indexCh].find("TrigEffAngle_Phi") == chambME[indexCh].end()){
	    bookChambHistos(chId,"TrigEffPosvsAngle_Phi");
	    bookChambHistos(chId,"TrigEffPosvsAngleHHHL_Phi");
	    bookChambHistos(chId,"TrigEffPos_Phi");
	    bookChambHistos(chId,"TrigEffPosHHHL_Phi");
	    bookChambHistos(chId,"TrigEffAngle_Phi");
	    bookChambHistos(chId,"TrigEffAngleHHHL_Phi");
	  }
	  if( secME[sector_id].find("TrigEffSec_Phi") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"TriggerAndSeg","TrigEffSec_Phi");  
	  }
	  if( whME[wh].find("TrigEff_Phi") == whME[wh].end() ){
	    bookWheelHistos(wh,"TriggerAndSeg","TrigEff_Phi");  
	    bookWheelHistos(wh,"TriggerAndSeg","TrigEffHHHL_Phi");  
	  }

	  std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	  TH1D* TrackPos               = TrackPosvsAngle->ProjectionY();
	  TH1D* TrackAngle             = TrackPosvsAngle->ProjectionX();
	  TH1D* TrackPosandTrig        = TrackPosvsAngleandTrig->ProjectionY();
	  TH1D* TrackAngleandTrig      = TrackPosvsAngleandTrig->ProjectionX();
	  TH1D* TrackPosandTrigHHHL    = TrackPosvsAngleandTrigHHHL->ProjectionY();
	  TH1D* TrackAngleandTrigHHHL  = TrackPosvsAngleandTrigHHHL->ProjectionX();
 	  float binEff     = float(TrackPosandTrig->GetEntries())/TrackPos->GetEntries();
 	  float binEffHHHL = float(TrackPosandTrigHHHL->GetEntries())/TrackPos->GetEntries();
 	  float binErr     = sqrt(binEff*(1-binEff)/TrackPos->GetEntries());
	  float binErrHHHL = sqrt(binEffHHHL*(1-binEffHHHL)/TrackPos->GetEntries());
	  
 	  MonitorElement* globalEff = innerME.find("TrigEffSec_Phi")->second;
 	  globalEff->setBinContent(stat,binEff);
 	  globalEff->setBinError(stat,binErr);

	  innerME = whME[wh];
	  int pos = stat+4*(sect-1);
	  globalEff = innerME.find("TrigEff_Phi")->second;
 	  globalEff->setBinContent(pos,binEff);
 	  globalEff->setBinError(pos,binErr);
	  globalEff = innerME.find("TrigEffHHHL_Phi")->second;
 	  globalEff->setBinContent(pos,binEffHHHL);
 	  globalEff->setBinError(pos,binErrHHHL);
	  
	  
	  innerME = chambME[indexCh];
	  makeEfficiencyME(TrackPosandTrig,TrackPos,innerME.find("TrigEffPos_Phi")->second);
	  makeEfficiencyME(TrackPosandTrigHHHL,TrackPos,innerME.find("TrigEffPosHHHL_Phi")->second);
	  makeEfficiencyME(TrackAngleandTrig,TrackAngle,innerME.find("TrigEffAngle_Phi")->second);
	  makeEfficiencyME(TrackAngleandTrigHHHL,TrackAngle,innerME.find("TrigEffAngleHHHL_Phi")->second);
	  makeEfficiencyME2D(TrackPosvsAngleandTrig,TrackPosvsAngle,innerME.find("TrigEffPosvsAngle_Phi")->second);
 	  makeEfficiencyME2D(TrackPosvsAngleandTrigHHHL,TrackPosvsAngle,innerME.find("TrigEffPosvsAngleHHHL_Phi")->second);
	     
	}
	
 	// Perform Efficiency analysis (Theta+Segments 2D)
	TH2F * TrackThetaPosvsAngle            = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngle","Segment", chId)));
	TH2F * TrackThetaPosvsAngleandTrig     = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngleandTrig","Segment", chId)));
	TH2F * TrackThetaPosvsAngleandTrigH    = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngleandTrigH","Segment", chId)));
	    
	if (TrackThetaPosvsAngle && TrackThetaPosvsAngleandTrig && TrackThetaPosvsAngleandTrigH) {
	      
 	  if( chambME[indexCh].find("TrigEffAngle_Theta") == chambME[indexCh].end()){
 	    bookChambHistos(chId,"TrigEffPosvsAngle_Theta");
 	    bookChambHistos(chId,"TrigEffPosvsAngleH_Theta");
	    bookChambHistos(chId,"TrigEffPos_Theta");
	    bookChambHistos(chId,"TrigEffPosH_Theta");
	    bookChambHistos(chId,"TrigEffAngle_Theta");
	    bookChambHistos(chId,"TrigEffAngleH_Theta");
	  }
	  if( secME[sector_id].find("TrigEffSec_Theta") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"TriggerAndSeg","TrigEffSec_Theta");  
	  }
	  if( whME[wh].find("TrigEff_Theta") == whME[wh].end() ){
	    bookWheelHistos(wh,"TriggerAndSeg","TrigEff_Theta");  
	    bookWheelHistos(wh,"TriggerAndSeg","TrigEffH_Theta");  
	  }

	  std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	  TH1D* TrackThetaPos               = TrackThetaPosvsAngle->ProjectionY();
	  TH1D* TrackThetaAngle             = TrackThetaPosvsAngle->ProjectionX();
	  TH1D* TrackThetaPosandTrig        = TrackThetaPosvsAngleandTrig->ProjectionY();
	  TH1D* TrackThetaAngleandTrig      = TrackThetaPosvsAngleandTrig->ProjectionX();
	  TH1D* TrackThetaPosandTrigH       = TrackThetaPosvsAngleandTrigH->ProjectionY();
	  TH1D* TrackThetaAngleandTrigH     = TrackThetaPosvsAngleandTrigH->ProjectionX();
	  float binEff  = float(TrackThetaPosandTrig->GetEntries())/TrackThetaPos->GetEntries();
 	  float binErr  = sqrt(binEff*(1-binEff)/TrackThetaPos->GetEntries());
 	  float binEffH = float(TrackThetaPosandTrigH->GetEntries())/TrackThetaPos->GetEntries();
 	  float binErrH = sqrt(binEffH*(1-binEffH)/TrackThetaPos->GetEntries());
 	  
 	  MonitorElement* globalEff = innerME.find("TrigEffSec_Theta")->second;
 	  globalEff->setBinContent(stat,binEff);
 	  globalEff->setBinError(stat,binErr);

	  innerME = whME[wh];
	  int pos = stat+3*(sect-1);
 	  globalEff = innerME.find("TrigEff_Theta")->second;
 	  globalEff->setBinContent(pos,binEff);
 	  globalEff->setBinError(pos,binErr);
	  globalEff = innerME.find("TrigEffH_Theta")->second;
 	  globalEff->setBinContent(pos,binEffH);
 	  globalEff->setBinError(pos,binErrH);
	  
	  innerME = chambME[indexCh];
	  makeEfficiencyME(TrackThetaPosandTrig,TrackThetaPos,innerME.find("TrigEffPos_Theta")->second);
	  makeEfficiencyME(TrackThetaPosandTrigH,TrackThetaPos,innerME.find("TrigEffPosH_Theta")->second);
	  makeEfficiencyME(TrackThetaAngleandTrig,TrackThetaAngle,innerME.find("TrigEffAngle_Theta")->second);
	  makeEfficiencyME(TrackThetaAngleandTrigH,TrackThetaAngle,innerME.find("TrigEffAngleH_Theta")->second);
       	  makeEfficiencyME2D(TrackThetaPosvsAngleandTrig,TrackThetaPosvsAngle,innerME.find("TrigEffPosvsAngle_Theta")->second);
  	  makeEfficiencyME2D(TrackThetaPosvsAngleandTrigH,TrackThetaPosvsAngle,innerME.find("TrigEffPosvsAngleH_Theta")->second);	     
	}

 	// Perform Correlation Plots analysis (DCC + segment Phi)
	TH2F * TrackPhitkvsPhitrig   = getHisto<TH2F>(dbe->get(getMEName("PhitkvsPhitrig","Segment", chId)));
	    
	if (TrackPhitkvsPhitrig) {
	      
	  // Fill client histos
	  if( secME[sector_id].find("PhiTkvsTrigCorr") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigSlope");  
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigIntercept");  
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigCorr");  
	  }

	  TProfile* PhitkvsPhitrigProf = TrackPhitkvsPhitrig->ProfileX();
	  PhitkvsPhitrigProf->Fit("pol1","Q");
	  TF1 *ffPhi= PhitkvsPhitrigProf->GetFunction("pol1");
	  double phiInt   = ffPhi->GetParameter(0);
	  double phiSlope = ffPhi->GetParameter(1);
	  double phiCorr  = TrackPhitkvsPhitrig->GetCorrelationFactor();

	  std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	  innerME.find("PhiTkvsTrigSlope")->second->setBinContent(stat,phiSlope);
	  innerME.find("PhiTkvsTrigIntercept")->second->setBinContent(stat,phiInt);
	  innerME.find("PhiTkvsTrigCorr")->second->setBinContent(stat,phiCorr);
	  
	}

 	// Perform Correlation Plots analysis (DCC + segment Phib)
	TH2F * TrackPhibtkvsPhibtrig = getHisto<TH2F>(dbe->get(getMEName("PhibtkvsPhibtrig","Segment", chId)));
	    
	if (stat != 3 && TrackPhibtkvsPhibtrig) {// station 3 has no meaningful MB3 phi bending information
	      
	  // Fill client histos
	  if( secME[sector_id].find("PhibTkvsTrigCorr") == secME[sector_id].end() ){
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigSlope");  
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigIntercept");  
	    bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigCorr");  
	  }

	  TProfile* PhibtkvsPhibtrigProf = TrackPhibtkvsPhibtrig->ProfileX(); 
	  PhibtkvsPhibtrigProf->Fit("pol1","Q");
	  TF1 *ffPhib= PhibtkvsPhibtrigProf->GetFunction("pol1");
	  double phibInt   = ffPhib->GetParameter(0);
	  double phibSlope = ffPhib->GetParameter(1);
	  double phibCorr  = TrackPhibtkvsPhibtrig->GetCorrelationFactor();

	  std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	  innerME.find("PhibTkvsTrigSlope")->second->setBinContent(stat,phibSlope);
	  innerME.find("PhibTkvsTrigIntercept")->second->setBinContent(stat,phibInt);
	  innerME.find("PhibTkvsTrigCorr")->second->setBinContent(stat,phibCorr);
	  
	}

      }
    }
  }	
  
  // Efficiency test (performed on chamber plots)
//   for(map<uint32_t,map<string,MonitorElement*> >::const_iterator imapIt = chambME.begin();
//       imapIt != chambME.end();
//       ++imapIt){
//     for (map<string,MonitorElement*>::const_iterator effME = (*imapIt).second.begin();
// 	 effME!=(*imapIt).second.end();
// 	 ++effME){
//       if ((*effME).second->getName().find("TrigEffPos_Phi") == 0) {
// 	const QReport *effQReport = (*effME).second->getQReport("ChambTrigEffInRangePhi");
// 	if (effQReport) {
// 	  if (effQReport->getBadChannels().size())
// 	    edm::LogError ("localTrigger") << (*effME).second->getName() <<" has " << effQReport->getBadChannels().size() << " channels out of expected efficiency range";
// 	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
// 	}
//       }
//       if ((*effME).second->getName().find("TrigEffPos_Theta") == 0) {
// 	const QReport *effQReport = (*effME).second->getQReport("ChambTrigEffInRangeTheta");
// 	if (effQReport) {
// 	  if (effQReport->getBadChannels().size())
// 	    edm::LogError ("localTrigger") << (*effME).second->getName() <<" has " << effQReport->getBadChannels().size() << " channels out of expected efficiency range";
// 	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
// 	}
//       }
//     }
//   }

  // Efficiency test (performed on wheel plots)
  for(map<int,map<string,MonitorElement*> >::const_iterator imapIt = secME.begin();
      imapIt != secME.end();
      ++imapIt){
    for (map<string,MonitorElement*>::const_iterator effME = (*imapIt).second.begin();
	 effME!=(*imapIt).second.end();
	 ++effME){
      if ((*effME).second->getName().find("TrigEffSec_Phi") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("SectorTrigEffInRangePhi");
	if (effQReport) {
	  // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
	  // edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
      if ((*effME).second->getName().find("TrigEffSec_Theta") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("SectorTrigEffInRangeTheta");
	if (effQReport) {
	  // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
	  // edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
    }
  }

}


void DTLocalTriggerTest::endJob(){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest] endjob called!";
  dbe->rmdir("DT/Tests/DTLocalTrigger");

}


void DTLocalTriggerTest::makeEfficiencyME(TH1D* numerator, TH1D* denominator, MonitorElement* result){
  
  TH1F* efficiency = result->getTH1F();
  efficiency->Divide(numerator,denominator,1,1,"");
  
  int nbins = efficiency->GetNbinsX();
  for (int bin=1; bin<=nbins; ++bin){
    float error = 0;
    float bineff = efficiency->GetBinContent(bin);

    if (denominator->GetBinContent(bin)){
      error = sqrt(bineff*(1-bineff)/denominator->GetBinContent(bin));
    }
    else {
      error = 1;
      efficiency->SetBinContent(bin,1.);
    }
 
    efficiency->SetBinError(bin,error);
  }

}


void DTLocalTriggerTest::makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result){
  
  TH2F* efficiency = result->getTH2F();
  efficiency->Divide(numerator,denominator,1,1,"");
  
  int nbinsx = efficiency->GetNbinsX();
  int nbinsy = efficiency->GetNbinsY();
  for (int binx=1; binx<=nbinsx; ++binx){
    for (int biny=1; biny<=nbinsy; ++biny){
      float error = 0;
      float bineff = efficiency->GetBinContent(binx,biny);

      if (denominator->GetBinContent(binx,biny)){
	error = sqrt(bineff*(1-bineff)/denominator->GetBinContent(binx,biny));
      }
      else {
	error = 1;
	efficiency->SetBinContent(binx,biny,0.);
      }
 
      efficiency->SetBinError(binx,biny,error);
    }
  }

}    


string DTLocalTriggerTest::getMEName(string histoTag, string subfolder, const DTChamberId & chambid) {

  stringstream wheel; wheel << chambid.wheel();
  stringstream station; station << chambid.station();
  stringstream sector; sector << chambid.sector();

//   if (subfolder == "Segment" && histoTag.find("Trig") == string::npos) 
//     histoTag = "SEG_" + histoTag;
//   else
  histoTag = hwSource + "_" + histoTag;

  string folderName = 
    "DT/DTLocalTriggerTask/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/" +  subfolder + "/";  

  string histoname = sourceFolder + folderName + histoTag  
    + "_W" + wheel.str()  
    + "_Sec" + sector.str()
    + "_St" + station.str();
  
  return histoname;
  
}

template <class T>
T* DTLocalTriggerTest::getHisto(MonitorElement* me) {
  return me ? dynamic_cast<T*>(me->getRootObject()) : 0;
}

void DTLocalTriggerTest::setLabelPh(MonitorElement* me){

  for (int i=0; i<48; ++i){
    stringstream label;
    int stat = (i%4) +1;
    if (stat==1) label << "Sec " << i/4 +1 << " ";
    me->setBinLabel(i+1,label.str().c_str());
  }

}

void DTLocalTriggerTest::setLabelTh(MonitorElement* me){

  for (int i=0; i<36; ++i){
    stringstream label;
    int stat = (i%3) +1;
    if (stat==1) label << "Sec " << i/3 +1 << " ";
    me->setBinLabel(i+1,label.str().c_str());
  }

}


void DTLocalTriggerTest::bookChambHistos(DTChamberId chambId, string htype) {
  
  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string HistoName = htype + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel" + wheel.str() +
			"/Sector" + sector.str() +
			"/Station" + station.str());
  
  uint32_t indexChId = chambId.rawId();
  if (htype.find("TrigEffAngle_Phi") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.);
    return;
  }
  if (htype.find("TrigEffAngleHHHL_Phi") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.);
    return;
  }
  if (htype.find("TrigEffAngle_Theta") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.);
    return;
  }
  if (htype.find("TrigEffAngleH_Theta") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.);
    return;
  }
  if (htype.find("TrigPosition_Phi") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),100,-500.,500.);
    return;
  }
  if (htype.find("TrigDirection_Phi") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),200,-40.,40.);
    return;
  }
  if (htype.find("TrigEffPos_Phi") == 0 ){
    pair<float,float> range = phiRange(chambId);
    int nbins = int((range.second - range.first)/15);
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),nbins,range.first,range.second);
    return;
  }
  if (htype.find("TrigEffPosvsAngle_Phi") == 0 ){
    pair<float,float> range = phiRange(chambId);
    int nbins = int((range.second - range.first)/15);
    chambME[indexChId][htype] = dbe->book2D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.,nbins,range.first,range.second);
    return;
  }
  if (htype.find("TrigEffPosvsAngleHHHL_Phi") == 0 ){
    pair<float,float> range = phiRange(chambId);
    int nbins = int((range.second - range.first)/15);
    chambME[indexChId][htype] = dbe->book2D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.,nbins,range.first,range.second);
    return;
  }
  if (htype.find("TrigEffPosHHHL_Phi") == 0 ){
    pair<float,float> range = phiRange(chambId);
    int nbins = int((range.second - range.first)/15);
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),nbins,range.first,range.second);
    return;
  }
  if (htype.find("TrigEffPos_Theta") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),20,-117.5,117.5);
    return;
  }
  if (htype.find("TrigEffPosH_Theta") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),20,-117.5,117.5);
    return;
  }
  if (htype.find("TrigEffPosvsAngle_Theta") == 0 ){
    chambME[indexChId][htype] = dbe->book2D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.,20,-117.5,117.5);
    return;
  }
  if (htype.find("TrigEffPosvsAngleH_Theta") == 0 ){
    chambME[indexChId][htype] = dbe->book2D(HistoName.c_str(),HistoName.c_str(),16,-40.,40.,20,-117.5,117.5);
    return;
  }

}


void DTLocalTriggerTest::bookSectorHistos(int wheel,int sector,string folder, string htype) {
  
  stringstream wh; wh << wheel;
  stringstream sc; sc << sector;
  int sectorid = (wheel+3) + (sector-1)*5;
  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel"+ wh.str()+"/Sector"+ sc.str()+"/"+folder);

  if (htype.find("Phi") != string::npos || 
      htype.find("TkvsTrig") != string::npos ){    
    string hname = htype + "_W" + wh.str()+"_Sec" +sc.str();
    MonitorElement* me = dbe->book1D(hname.c_str(),hname.c_str(),4,1,5);
    secME[sectorid][htype] = me;
    return;
  }
  
  if (htype.find("Theta") != string::npos){
    string hname = htype + "_W" + wh.str()+ "_Sec"+sc.str();
    MonitorElement* me =dbe->book1D(hname.c_str(),hname.c_str(),3,1,4);
    secME[sectorid][htype] = me;
    return;
  }
  
}

void DTLocalTriggerTest::bookWheelHistos(int wheel, string folder, string htype) {
  
  stringstream wh; wh << wheel;
  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel"+ wh.str()+"/"+folder);

  if (htype.find("Phi") != string::npos){    
    string hname = htype + "_W" + wh.str();
    MonitorElement* me = dbe->book1D(hname.c_str(),hname.c_str(),48,1,49);
    setLabelPh(me);
    whME[wheel][htype] = me;
    return;
  }
  
  if (htype.find("Theta") != string::npos){
    string hname = htype + "_W" + wh.str();
    MonitorElement* me =dbe->book1D(hname.c_str(),hname.c_str(),36,1,37);
    setLabelTh(me);
    whME[wheel][htype] = me;
    return;
  }
  
}

pair<float,float> DTLocalTriggerTest::phiRange(const DTChamberId& id){

  float min,max;
  int station = id.station();
  int sector  = id.sector(); 
  int wheel   = id.wheel();
  
  const DTLayer  *layer = muonGeom->layer(DTLayerId(id,1,1));
  DTTopology topo = layer->specificTopology();
  min = topo.wirePosition(topo.firstChannel());
  max = topo.wirePosition(topo.lastChannel());

  if (station == 4){
    
    const DTLayer *layer2;
    float lposx;
    
    if (sector == 4){
      layer2  = muonGeom->layer(DTLayerId(wheel,station,13,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else if (sector == 10){
      layer2 = muonGeom->layer(DTLayerId(wheel,station,14,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else
      return make_pair(min,max);
    
    DTTopology topo2 = layer2->specificTopology();

    if (lposx>0){
      max = lposx*.5+topo2.wirePosition(topo2.lastChannel());
      min -= lposx*.5;
    }
    else{
      min = lposx*.5+topo2.wirePosition(topo2.firstChannel());
      max -= lposx*.5;
    }
      
  }
  
  return make_pair(min,max);

}
