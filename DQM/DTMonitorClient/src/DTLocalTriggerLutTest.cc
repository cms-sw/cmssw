/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/05 08:00:50 $
 *  $Revision: 1.3 $
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


void DTLocalTriggerLutTest::beginJob(const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginJob(c);


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
	    bookSectorHistos(wh,sect,"","PhiTkvsTrigSlope");  
	    bookSectorHistos(wh,sect,"","PhiTkvsTrigIntercept");  
	    bookSectorHistos(wh,sect,"","PhiTkvsTrigCorr");  
	    bookSectorHistos(wh,sect,"","PhibTkvsTrigSlope");  
	    bookSectorHistos(wh,sect,"","PhibTkvsTrigIntercept");  
	    bookSectorHistos(wh,sect,"","PhibTkvsTrigCorr");  
	  }
	  bookWheelHistos(wh,"","PhiTkvsTrigSlope");  
	  bookWheelHistos(wh,"","PhiTkvsTrigIntercept");  
	  bookWheelHistos(wh,"","PhiTkvsTrigCorr");  
	  bookWheelHistos(wh,"","PhibTkvsTrigSlope");  
	  bookWheelHistos(wh,"","PhibTkvsTrigIntercept");  
	  bookWheelHistos(wh,"","PhibTkvsTrigCorr");  
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
	bookWheelHistos(wh,"","PhiSlopeSummary");
	bookWheelHistos(wh,"","PhibSlopeSummary");
      }
      bookCmsHistos("PhiSlopeSummary");
      bookCmsHistos("PhibSlopeSummary");
    }	
  }

}


void DTLocalTriggerLutTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  edm::LogVerbatim ("localTrigger") <<"[" << testName << "Test]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs++;

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
	    
	    if (TrackPhitkvsPhitrig && TrackPhitkvsPhitrig->GetEntries()>10) {
	      
	      // Fill client histos
	      if( secME[sector_id].find(fullName("PhiTkvsTrigCorr")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"","PhiTkvsTrigSlope");  
		bookSectorHistos(wh,sect,"","PhiTkvsTrigIntercept");  
		bookSectorHistos(wh,sect,"","PhiTkvsTrigCorr");  
	      }
	      if( whME[wh].find(fullName("PhiTkvsTrigCorr")) == whME[wh].end() ){
		bookWheelHistos(wh,"","PhiTkvsTrigSlope");  
		bookWheelHistos(wh,"","PhiTkvsTrigIntercept");  
		bookWheelHistos(wh,"","PhiTkvsTrigCorr");  
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
	    
	    if (stat != 3 && TrackPhibtkvsPhibtrig && TrackPhibtkvsPhibtrig->GetEntries()>10) {// station 3 has no meaningful MB3 phi bending information
	      
	      // Fill client histos
	      if( secME[sector_id].find(fullName("PhibTkvsTrigCorr")) == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"","PhibTkvsTrigSlope");  
		bookSectorHistos(wh,sect,"","PhibTkvsTrigIntercept");  
		bookSectorHistos(wh,sect,"","PhibTkvsTrigCorr");  
	      }
	      if( whME[wh].find(fullName("PhibTkvsTrigCorr")) == whME[wh].end() ){
 		bookWheelHistos(wh,"","PhibTkvsTrigSlope");  
		bookWheelHistos(wh,"","PhibTkvsTrigIntercept");  
		bookWheelHistos(wh,"","PhibTkvsTrigCorr");  
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
	
  // Summary Plots
  map<int,map<string,MonitorElement*> >::const_iterator imapIt = secME.begin();
  map<int,map<string,MonitorElement*> >::const_iterator mapEnd = secME.end();

  for(; imapIt != mapEnd; ++imapIt){
    int sector = ((*imapIt).first-1)/5 + 1;
    int wheel  = ((*imapIt).first-1)%5 - 2;

    for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
      trigSource = (*iTr);
      for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
	hwSource = (*iHw);

	MonitorElement *testME = (*imapIt).second.find(fullName("PhiTkvsTrigSlope"))->second;
	bool hasEntries = testME->getEntries()>=1;
	const QReport *testQReport = testME->getQReport("TrigPhiSlopeInRange");

	if (testQReport) {
	  int err = testQReport->getBadChannels().size();	  
	  if (err<0 || err>4) err=4;
	  cmsME.find(fullName("PhiSlopeSummary"))->second->setBinContent(sector,wheel+3,err);
	  vector<dqm::me_util::Channel> badChannels = testQReport->getBadChannels();
	  int cherr[4];
	  for (int i=0;i<4;++i) 
	    cherr[i] = hasEntries ? 0 : 1;
	  if (hasEntries) {
	    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
		 channel != badChannels.end(); channel++) {
	      cherr[(*channel).getBin()-1] = 2 ; 
	    }
	  }
	  for (int i=0;i<4;++i)
	    whME.find(wheel)->second.find(fullName("PhiSlopeSummary"))->second->setBinContent(sector,i+1,cherr[i]);
	}

	testME = (*imapIt).second.find(fullName("PhibTkvsTrigSlope"))->second;
	hasEntries = testME->getEntries()>=1;
	testQReport = testME->getQReport("TrigPhibSlopeInRange");

	if (testQReport) {
	  int err = testQReport->getBadChannels().size();	  
	  if (err<0 || err>4) err=4;
	  vector<dqm::me_util::Channel> badChannels = testQReport->getBadChannels();
	  int cherr[4];
	  for (int i=0;i<4;++i)
	    cherr[i] = hasEntries ? 0 : 1;
	  if (hasEntries) {
	    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
		 channel != badChannels.end(); channel++) {
	      cherr[(*channel).getBin()-1] = 2 ;
	      if ((*channel).getBin()==3) err-=1;
	    }
	  }
	  cherr[2] = 1; //MB3 has no meaningful phib info!
	  cmsME.find(fullName("PhibSlopeSummary"))->second->setBinContent(sector,wheel+3,err);
	  for (int i=0;i<4;++i)
	    whME.find(wheel)->second.find(fullName("PhibSlopeSummary"))->second->setBinContent(sector,i+1,cherr[i]);
	}
      }
    }
  }
}

