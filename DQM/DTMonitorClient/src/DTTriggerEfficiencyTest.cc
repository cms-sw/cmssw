/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/04/09 15:45:24 $
 *  $Revision: 1.7 $
 *  \author C. Battilana - CIEMAT
 */


// This class header
#include "DQM/DTMonitorClient/src/DTTriggerEfficiencyTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// Trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTTriggerEfficiencyTest::DTTriggerEfficiencyTest(const edm::ParameterSet& ps){

  setConfig(ps,"DTTriggerEfficiency");
  baseFolderDCC = "DT/03-LocalTrigger-DCC/";
  baseFolderDDU = "DT/04-LocalTrigger-DDU/";
  detailedPlots = ps.getUntrackedParameter<bool>("detailedAnalysis",true);

}


DTTriggerEfficiencyTest::~DTTriggerEfficiencyTest(){
  
}


void DTTriggerEfficiencyTest::beginJob(const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginJob(c);
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
	  if (detailedPlots) {
	    for (int sect=1; sect<=12; ++sect){
	      for (int stat=1; stat<=4; ++stat){
		DTChamberId chId(wh,stat,sect);
		bookChambHistos(chId,"TrigEffPosvsAnglePhi","Segment");
		bookChambHistos(chId,"TrigEffPosvsAngleCorrPhi","Segment");
	      }
	    }
	  }
	  bookWheelHistos(wh,"","TrigEffPhi");  
	  bookWheelHistos(wh,"","TrigEffCorrPhi");  
	}
      }
    }
  }
  
}


void DTTriggerEfficiencyTest::runClientDiagnostic() {

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      for (int wh=-2; wh<=2; ++wh){
	
	TH2F * TrigEffDenum   = getHisto<TH2F>(dbe->get(getMEName("TrigEffDenum","",wh)));
	TH2F * TrigEffNum     = getHisto<TH2F>(dbe->get(getMEName("TrigEffNum","",wh)));
	TH2F * TrigEffCorrNum = getHisto<TH2F>(dbe->get(getMEName("TrigEffCorrNum","",wh)));
	
	if (TrigEffDenum && TrigEffNum && TrigEffCorrNum && TrigEffDenum->GetEntries()>1) {
	  
	  if( whME[wh].find(fullName("TrigEffPhi")) == whME[wh].end() ){
	    bookWheelHistos(wh,"","TrigEffPhi");  
	    bookWheelHistos(wh,"","TrigEffCorrPhi");  
	  }
	  std::map<std::string,MonitorElement*> *innerME = &(whME[wh]);
	  makeEfficiencyME2D(TrigEffNum,TrigEffDenum,innerME->find(fullName("TrigEffPhi"))->second);
	  makeEfficiencyME2D(TrigEffCorrNum,TrigEffDenum,innerME->find(fullName("TrigEffCorrPhi"))->second);
	  
	}

	if (detailedPlots) {
	  for (int stat=1; stat<=4; ++stat){
	    for (int sect=1; sect<=12; ++sect){
	      DTChamberId chId(wh,stat,sect);
	      uint32_t indexCh = chId.rawId();
	      
	      // Perform Efficiency analysis (Phi+Segments 2D)
	      TH2F * TrackPosvsAngle        = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngle","Segment", chId)));
	      TH2F * TrackPosvsAngleAnyQual = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleAnyQual","Segment", chId)));
	      TH2F * TrackPosvsAngleCorr    = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleCorr","Segment", chId)));
	    
	      if (TrackPosvsAngle && TrackPosvsAngleAnyQual && TrackPosvsAngleCorr && TrackPosvsAngle->GetEntries()>1) {
	      
		if( chambME[indexCh].find(fullName("TrigEffAnglePhi")) == chambME[indexCh].end()){
		  bookChambHistos(chId,"TrigEffPosvsAnglePhi","Segment");
		  bookChambHistos(chId,"TrigEffPosvsAngleCorrPhi","Segment");
		}
		
		std::map<std::string,MonitorElement*> *innerME = &(chambME[indexCh]);
		makeEfficiencyME2D(TrackPosvsAngleAnyQual,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAnglePhi"))->second);
		makeEfficiencyME2D(TrackPosvsAngleCorr,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAngleCorrPhi"))->second);
	     
	      }
	    }
	  }
	}
      }

    }
  }	

}

void DTTriggerEfficiencyTest::makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result){
  
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


void DTTriggerEfficiencyTest::bookChambHistos(DTChamberId chambId, string htype, string folder) {
  
  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string fullType  = fullName(htype);
  bool isDCC = hwSource=="DCC" ;
  string HistoName = fullType + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder(topFolder(isDCC) + 
			"Wheel" + wheel.str() +
			"/Sector" + sector.str() +
			"/Station" + station.str() + 
			"/" + folder + "/");
  
  LogTrace(category()) << "[" << testName << "Test]: booking " + topFolder(isDCC) + "Wheel" << wheel.str() 
		       <<"/Sector" << sector.str() << "/Station" << station.str() << "/" + folder + "/" << HistoName;

  
  uint32_t indexChId = chambId.rawId();
  if (htype.find("TrigEffPosvsAnglePhi") == 0 ){
    float min, max;
    trigGeomUtils->phiRange(chambId,min,max);
    int nbins = int((max- min)/20);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (any qual.) position vs angle (Phi)",12,-30.,30.,nbins,min,max);
    return;
  }
  if (htype.find("TrigEffPosvsAngleCorrPhi") == 0 ){
    float min, max;
    trigGeomUtils->phiRange(chambId,min,max);
    int nbins = int((max- min)/20);
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (correlated) pos vs angle (Phi)",12,-30.,30.,nbins,min,max);
    return;
  }

}
