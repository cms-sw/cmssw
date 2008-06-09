/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/01 00:39:51 $
 *  $Revision: 1.15 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerLutTest.h"

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


DTLocalTriggerLutTest::DTLocalTriggerLutTest(const edm::ParameterSet& ps){

  setConfig(ps,"DTLocalTriggerLut");

}


DTLocalTriggerLutTest::~DTLocalTriggerLutTest(){

}


void DTLocalTriggerLutTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
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

	    // Perform Correlation Plots analysis (DCC + segment Phi)
	    TH2F * TrackPhitkvsPhitrig   = getHisto<TH2F>(dbe->get(getMEName("PhitkvsPhitrig","Segment", chId)));
	    
	    if (TrackPhitkvsPhitrig) {
	      
	      // Fill client histos
	      if( secME[sector_id].find(fullName("PhiTkvsTrigCorr")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigSlope");  
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigIntercept");  
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhiTkvsTrigCorr");  
	      }
	      if( whME[wh].find(fullName("PhiTkvsTrigCorr")) == whME[wh].end() ){
		bookWheelHistos(wh,"TriggerAndSeg","PhiTkvsTrigSlope");  
		bookWheelHistos(wh,"TriggerAndSeg","PhiTkvsTrigIntercept");  
		bookWheelHistos(wh,"TriggerAndSeg","PhiTkvsTrigCorr");  
	      }

	      TProfile* PhitkvsPhitrigProf = TrackPhitkvsPhitrig->ProfileX();
	      PhitkvsPhitrigProf->Fit("pol1","CQO");
	      TF1 *ffPhi= PhitkvsPhitrigProf->GetFunction("pol1");
	      double phiInt   = ffPhi->GetParameter(0);
	      double phiSlope = ffPhi->GetParameter(1);
	      double phiCorr  = TrackPhitkvsPhitrig->GetCorrelationFactor();

	      std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
	      innerME->find(fullName("PhiTkvsTrigSlope"))->second->setBinContent(stat,phiSlope);
	      innerME->find(fullName("PhiTkvsTrigIntercept"))->second->setBinContent(stat,phiInt);
	      innerME->find(fullName("PhiTkvsTrigCorr"))->second->setBinContent(stat,phiCorr);

	      innerME = &(whME[wh]);
	      innerME->find(fullName("PhiTkvsTrigSlope"))->second->setBinContent(sect,stat,phiSlope);
	      innerME->find(fullName("PhiTkvsTrigIntercept"))->second->setBinContent(sect,stat,phiInt);
	      innerME->find(fullName("PhiTkvsTrigCorr"))->second->setBinContent(sect,stat,phiCorr);

	    }

	    // Perform Correlation Plots analysis (DCC + segment Phib)
	    TH2F * TrackPhibtkvsPhibtrig = getHisto<TH2F>(dbe->get(getMEName("PhibtkvsPhibtrig","Segment", chId)));
	    
	    if (stat != 3 && TrackPhibtkvsPhibtrig) {// station 3 has no meaningful MB3 phi bending information
	      
	      // Fill client histos
	      if( secME[sector_id].find(fullName("PhibTkvsTrigCorr")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigSlope");  
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigIntercept");  
		bookSectorHistos(wh,sect,"TriggerAndSeg","PhibTkvsTrigCorr");  
	      }
	      if( whME[wh].find(fullName("PhibTkvsTrigCorr")) == whME[wh].end() ){
		bookWheelHistos(wh,"TriggerAndSeg","PhibTkvsTrigSlope");  
		bookWheelHistos(wh,"TriggerAndSeg","PhibTkvsTrigIntercept");  
		bookWheelHistos(wh,"TriggerAndSeg","PhibTkvsTrigCorr");  
	      }

	      TProfile* PhibtkvsPhibtrigProf = TrackPhibtkvsPhibtrig->ProfileX(); 
	      PhibtkvsPhibtrigProf->Fit("pol1","CQO");
	      TF1 *ffPhib= PhibtkvsPhibtrigProf->GetFunction("pol1");
	      double phibInt   = ffPhib->GetParameter(0);
	      double phibSlope = ffPhib->GetParameter(1);
	      double phibCorr  = TrackPhibtkvsPhibtrig->GetCorrelationFactor();

	      std::map<std::string,MonitorElement*> *innerME = &(secME[sector_id]);
	      innerME->find(fullName("PhibTkvsTrigSlope"))->second->setBinContent(stat,phibSlope);
	      innerME->find(fullName("PhibTkvsTrigIntercept"))->second->setBinContent(stat,phibInt);
	      innerME->find(fullName("PhibTkvsTrigCorr"))->second->setBinContent(stat,phibCorr);
	  
	      innerME = &(whME[wh]);
	      innerME->find(fullName("PhibTkvsTrigSlope"))->second->setBinContent(sect,stat,phibSlope);
	      innerME->find(fullName("PhibTkvsTrigIntercept"))->second->setBinContent(sect,stat,phibInt);
	      innerME->find(fullName("PhibTkvsTrigCorr"))->second->setBinContent(sect,stat,phibCorr);
	  
	    }

	  }
	}
      }
    }
  }	
  
}

