/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.10 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerEfficiencyTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"
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


DTLocalTriggerEfficiencyTest::DTLocalTriggerEfficiencyTest(const edm::ParameterSet& ps) : trigGeomUtils(0) {

  setConfig(ps,"DTLocalTriggerEfficiency");
  baseFolderDCC = "DT/03-LocalTrigger-DCC/";
  baseFolderDDU = "DT/04-LocalTrigger-DDU/";

}


DTLocalTriggerEfficiencyTest::~DTLocalTriggerEfficiencyTest(){

  if (trigGeomUtils) {
    delete trigGeomUtils;
  }
  
}


void DTLocalTriggerEfficiencyTest::beginRun(const edm::Run& r, const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginRun(r,c);
  trigGeomUtils = new DTTrigGeomUtils(muonGeom);

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
	  for (int sect=1; sect<=12; ++sect){
	    for (int stat=1; stat<=4; ++stat){
	      DTChamberId chId(wh,stat,sect);
	      bookChambHistos(chId,"TrigEffPosvsAnglePhi");
	      bookChambHistos(chId,"TrigEffPosvsAngleHHHLPhi");
	      bookChambHistos(chId,"TrigEffPosPhi");
	      bookChambHistos(chId,"TrigEffPosHHHLPhi");
	      bookChambHistos(chId,"TrigEffAnglePhi");
	      bookChambHistos(chId,"TrigEffAngleHHHLPhi");
	      if (stat<=3) {
		bookChambHistos(chId,"TrigEffPosvsAngleTheta");
		bookChambHistos(chId,"TrigEffPosvsAngleHTheta");
		bookChambHistos(chId,"TrigEffPosTheta");
		bookChambHistos(chId,"TrigEffPosHTheta");
		bookChambHistos(chId,"TrigEffAngleTheta");
		bookChambHistos(chId,"TrigEffAngleHTheta");
	      }
	    }
	    bookSectorHistos(wh,sect,"TrigEffPhi");  
	    bookSectorHistos(wh,sect,"TrigEffTheta");  
	  }
	  bookWheelHistos(wh,"TrigEffPhi");  
	  bookWheelHistos(wh,"TrigEffHHHLPhi");  
	  bookWheelHistos(wh,"TrigEffTheta");  
	  bookWheelHistos(wh,"TrigEffHTheta");  
	}
      }
    }
  }
  
}


void DTLocalTriggerEfficiencyTest::runClientDiagnostic() {

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
	    uint32_t indexCh = chId.rawId();

	    // Perform Efficiency analysis (Phi+Segments 2D)
	    TH2F * TrackPosvsAngle            = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngle","Segment", chId)));
	    TH2F * TrackPosvsAngleandTrig     = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleandTrig","Segment", chId)));
	    TH2F * TrackPosvsAngleandTrigHHHL = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleandTrigHHHL","Segment", chId)));
	    
	    if (TrackPosvsAngle && TrackPosvsAngleandTrig && TrackPosvsAngleandTrigHHHL && TrackPosvsAngle->GetEntries()>1) {
	      
	      if( chambME[indexCh].find(fullName("TrigEffAnglePhi")) == chambME[indexCh].end()){
		bookChambHistos(chId,"TrigEffPosvsAnglePhi");
		bookChambHistos(chId,"TrigEffPosvsAngleHHHLPhi");
		bookChambHistos(chId,"TrigEffPosPhi");
		bookChambHistos(chId,"TrigEffPosHHHLPhi");
		bookChambHistos(chId,"TrigEffAnglePhi");
		bookChambHistos(chId,"TrigEffAngleHHHLPhi");
	      }
	      if( secME[sector_id].find(fullName("TrigEffPhi")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TrigEffPhi");  
	      }
	      if( whME[wh].find(fullName("TrigEffPhi")) == whME[wh].end() ){
		bookWheelHistos(wh,"TrigEffPhi");  
		bookWheelHistos(wh,"TrigEffHHHLPhi");  
	      }

	      std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
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
	  
	      MonitorElement* globalEff = innerME->find(fullName("TrigEffPhi"))->second;
	      globalEff->setBinContent(stat,binEff);
	      globalEff->setBinError(stat,binErr);

	      innerME = &(whME[wh]);
	      globalEff = innerME->find(fullName("TrigEffPhi"))->second;
	      globalEff->setBinContent(sect,stat,binEff);
	      globalEff->setBinError(sect,stat,binErr);
	      globalEff = innerME->find(fullName("TrigEffHHHLPhi"))->second;
	      globalEff->setBinContent(sect,stat,binEffHHHL);
	      globalEff->setBinError(sect,stat,binErrHHHL);
	  
	  
	      innerME = &(chambME[indexCh]);
	      makeEfficiencyME(TrackPosandTrig,TrackPos,innerME->find(fullName("TrigEffPosPhi"))->second);
	      makeEfficiencyME(TrackPosandTrigHHHL,TrackPos,innerME->find(fullName("TrigEffPosHHHLPhi"))->second);
	      makeEfficiencyME(TrackAngleandTrig,TrackAngle,innerME->find(fullName("TrigEffAnglePhi"))->second);
	      makeEfficiencyME(TrackAngleandTrigHHHL,TrackAngle,innerME->find(fullName("TrigEffAngleHHHLPhi"))->second);
	      makeEfficiencyME2D(TrackPosvsAngleandTrig,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAnglePhi"))->second);
	      makeEfficiencyME2D(TrackPosvsAngleandTrigHHHL,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAngleHHHLPhi"))->second);
	     
	    }
	
	    // Perform Efficiency analysis (Theta+Segments)  CB FIXME -> no DCC theta qual info
	    TH2F * TrackThetaPosvsAngle            = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngle","Segment", chId)));
	    TH2F * TrackThetaPosvsAngleandTrig     = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngleandTrig","Segment", chId)));
	    TH2F * TrackThetaPosvsAngleandTrigH    = getHisto<TH2F>(dbe->get(getMEName("TrackThetaPosvsAngleandTrigH","Segment", chId)));
	    
	    if (TrackThetaPosvsAngle && TrackThetaPosvsAngleandTrig && TrackThetaPosvsAngleandTrigH && TrackThetaPosvsAngle->GetEntries()>1) {
	      
	      if( chambME[indexCh].find(fullName("TrigEffAngleTheta")) == chambME[indexCh].end()){
		bookChambHistos(chId,"TrigEffPosvsAngleTheta");
		bookChambHistos(chId,"TrigEffPosvsAngleHTheta");
		bookChambHistos(chId,"TrigEffPosTheta");
		bookChambHistos(chId,"TrigEffPosHTheta");
		bookChambHistos(chId,"TrigEffAngleTheta");
		bookChambHistos(chId,"TrigEffAngleHTheta");
	      }
	      if( secME[sector_id].find(fullName("TrigEffTheta")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TrigEffTheta");  
	      }
	      if( whME[wh].find(fullName("TrigEffTheta")) == whME[wh].end() ){
		bookWheelHistos(wh,"TrigEffTheta");  
		bookWheelHistos(wh,"TrigEffHTheta");  
	      }

	      std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
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
 	  
	      MonitorElement* globalEff = innerME->find(fullName("TrigEffTheta"))->second;
	      globalEff->setBinContent(stat,binEff);
	      globalEff->setBinError(stat,binErr);

	      innerME = &(whME[wh]);
	      globalEff = innerME->find(fullName("TrigEffTheta"))->second;
	      globalEff->setBinContent(sect,stat,binEff);
	      globalEff->setBinError(sect,stat,binErr);
	      globalEff = innerME->find(fullName("TrigEffHTheta"))->second;
	      globalEff->setBinContent(sect,stat,binEffH);
	      globalEff->setBinError(sect,stat,binErrH);
	  
	      innerME = &(chambME[indexCh]);
	      makeEfficiencyME(TrackThetaPosandTrig,TrackThetaPos,innerME->find(fullName("TrigEffPosTheta"))->second);
	      makeEfficiencyME(TrackThetaPosandTrigH,TrackThetaPos,innerME->find(fullName("TrigEffPosHTheta"))->second);
	      makeEfficiencyME(TrackThetaAngleandTrig,TrackThetaAngle,innerME->find(fullName("TrigEffAngleTheta"))->second);
	      makeEfficiencyME(TrackThetaAngleandTrigH,TrackThetaAngle,innerME->find(fullName("TrigEffAngleHTheta"))->second);
	      makeEfficiencyME2D(TrackThetaPosvsAngleandTrig,TrackThetaPosvsAngle,innerME->find(fullName("TrigEffPosvsAngleTheta"))->second);
	      makeEfficiencyME2D(TrackThetaPosvsAngleandTrigH,TrackThetaPosvsAngle,innerME->find(fullName("TrigEffPosvsAngleHTheta"))->second);	     
	    }

	  }
	}
      }
    }
  }	

}


void DTLocalTriggerEfficiencyTest::makeEfficiencyME(TH1D* numerator, TH1D* denominator, MonitorElement* result){
  
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


void DTLocalTriggerEfficiencyTest::makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result){
  
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


void DTLocalTriggerEfficiencyTest::bookChambHistos(DTChamberId chambId, string htype) {
  
  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string fullType  = fullName(htype);
  bool isDCC = hwSource=="DCC" ;
  string HistoName = fullType + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder(topFolder(isDCC) + "Wheel" + wheel.str() +
			"/Sector" + sector.str() +
			"/Station" + station.str() + "/Segment");

  LogTrace(category()) << "[" << testName << "Test]: booking " + topFolder(isDCC) + "Wheel" << wheel.str() 
		       <<"/Sector" << sector.str() << "/Station" << station.str() << "/Segment/" << HistoName;

  
  uint32_t indexChId = chambId.rawId();
  if (htype.find("TrigEffAnglePhi") == 0){
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency vs angle of incidence (Phi)",16,-40.,40.);
  }
  else if (htype.find("TrigEffAngleHHHLPhi") == 0){
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency (HH/HL) vs angle of incidence (Phi)",16,-40.,40.);
  }
  else if (htype.find("TrigEffAngleTheta") == 0){
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency vs angle of incidence (Theta)",16,-40.,40.);
  }
  else if (htype.find("TrigEffAngleHTheta") == 0){
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency (H) vs angle of incidence (Theta)",16,-40.,40.);
  }
  else if (htype.find("TrigEffPosPhi") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->phiRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency vs position (Phi)",nbins,min,max);
  }
  else if (htype.find("TrigEffPosvsAnglePhi") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->phiRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency position vs angle (Phi)",16,-40.,40.,nbins,min,max);
  }
  else if (htype.find("TrigEffPosvsAngleHHHLPhi") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->phiRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (HH/HL) pos vs angle (Phi)",16,-40.,40.,nbins,min,max);
  }
  else if (htype.find("TrigEffPosHHHLPhi") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->phiRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency (HH/HL) vs position (Phi)",nbins,min,max);
  }
  else if (htype.find("TrigEffPosTheta") == 0){
    float min,max;
    int nbins;
    trigGeomUtils->thetaRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency vs position (Theta)",nbins,min,max);
  }
  else if (htype.find("TrigEffPosHTheta") == 0){
    float min,max;
    int nbins;
    trigGeomUtils->thetaRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"Trigger efficiency (H) vs position (Theta)",nbins,min,max);
  }
  else if (htype.find("TrigEffPosvsAngleTheta") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->thetaRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency pos vs angle (Theta)",16,-40.,40.,nbins,min,max);
  }
  else if (htype.find("TrigEffPosvsAngleHTheta") == 0 ){
    float min,max;
    int nbins;
    trigGeomUtils->thetaRange(chambId,min,max,nbins);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (H) pos vs angle (Theta)",16,-40.,40.,nbins,min,max);
  }

}
