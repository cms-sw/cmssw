/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.4 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerTPTest.h"

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


DTLocalTriggerTPTest::DTLocalTriggerTPTest(const edm::ParameterSet& ps){

  setConfig(ps,"DTLocalTriggerTP");
  baseFolderDCC = "DT/11-LocalTriggerTP-DCC/";
  baseFolderDDU = "DT/12-LocalTriggerTP-DDU/";
  

}


DTLocalTriggerTPTest::~DTLocalTriggerTPTest(){

}

void DTLocalTriggerTPTest::beginJob(){
  
  DTLocalTriggerBaseTest::beginJob();


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
	  bookWheelHistos(wh,"CorrectBXPhi");
	  bookWheelHistos(wh,"ResidualBXPhi");
	}
      }
    }
  }

}


void DTLocalTriggerTPTest::beginRun(const edm::Run& r, const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginRun(r,c);

}


void DTLocalTriggerTPTest::runClientDiagnostic() {

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
	    // int sector_id = (wh+3)+(sect-1)*5;
	    // uint32_t indexCh = chId.rawId();
	    

	    // Perform DCC/DDU common plot analysis (Phi ones)
	    TH2F * BXvsQual      = getHisto<TH2F>(dbe->get(getMEName("BXvsQual","LocalTriggerPhi", chId)));
	    if ( BXvsQual ) {

	      if (BXvsQual->GetEntries()>1) {
	      
		TH1D* BX    = BXvsQual->ProjectionY();
		int BXOK_bin  = BX->GetMaximumBin();
		double BXMean = BX->GetMean();
		double BX_OK  = BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
		delete BX;

		if( whME[wh].find(fullName("CorrectBXPhi")) == whME[wh].end() ){
		  bookWheelHistos(wh,"ResidualBXPhi");
		  bookWheelHistos(wh,"CorrectBXPhi");
		}
	   
		std::map<std::string,MonitorElement*> *innerME = &(whME[wh]);
		innerME->find(fullName("CorrectBXPhi"))->second->setBinContent(sect,stat,BX_OK+0.00001);
		innerME->find(fullName("ResidualBXPhi"))->second->setBinContent(sect,stat,round(25.*(BXMean-BX_OK))+0.00001);
	      }
	      
	    }
	  }
	}
      }
    }
  }	

}

