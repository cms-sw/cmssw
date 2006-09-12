/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/09/07 15:30:32 $
 *  $Revision: 1.0 $
 *  \author S. Bolognesi
 */

#include "CalibMuon/DTCalibration/src/DTTTrigWriter.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

/* C++ Headers */
#include <vector> 
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1.h"

using namespace std;
using namespace edm;


DTTTrigWriter::DTTTrigWriter(const ParameterSet& pset) {
  // get selected debug option
  debug = pset.getUntrackedParameter<bool>("debug", "false");

  // Open the root file which contains the histos
  theRootInputFile = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(theRootInputFile.c_str(), "READ");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);

 // Create the object to be written to DB
  string tag = pset.getUntrackedParameter<string>("tTrigTag", "ttrig_test");
  tTrig = new DTTtrig(tag);
  
  if(debug)
    cout << "[DTTTrigWriter]Constructor called!" << endl;
}

DTTTrigWriter::~DTTTrigWriter(){
  if(debug)
    cout << "[DTTTrigWriter]Destructor called!" << endl;
}

void DTTTrigWriter::analyze(const Event & event, const EventSetup& eventSetup) {
  if(debug)
    cout << "[DTTTrigWriter]Analyzer called!" << endl;

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get all the sls from the setup
  const vector<DTSuperLayer*> superLayers = dtGeom->superLayers(); 
    
  // Loop over all SLs
  for(vector<DTSuperLayer*>::const_iterator  sl = superLayers.begin();
      sl != superLayers.end(); sl++) {
      
    //Get the histo from file
    DTSuperLayerId slId = (*sl)->id();
    //Compute mean and sigma of the rising edge
    pair<double, double> meanAndSigma = theFitter->fitTimeBox((TH1F*)theFile->Get((getTBoxName(slId)).c_str()));

    //Write them in DB object
    tTrig->setSLTtrig(slId,
		      meanAndSigma.first,
		      meanAndSigma.second,
		      DTTimeUnits::ns);
    if(debug) {
      cout << " SL: " << slId
	   << " mean = " << meanAndSigma.first
	   << " sigma = " << meanAndSigma.second << endl;
    }
  }
}

void DTTTrigWriter::endJob() {
  
  if(debug) 
	cout << "[DTTTrigWriter]Writing ttrig object to DB!" << endl;
 
  // Write the ttrig object to DB
  edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
  if( dbOutputSvc.isAvailable() ){
    //size_t callbackToken = dbOutputSvc->callbackToken("DTTtrig");
    size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    try{
      dbOutputSvc->newValidityForNewPayload<DTTtrig>(tTrig, dbOutputSvc->endOfTime(), callbackToken);
    }catch(const cond::Exception& er){
      cout << er.what() << endl;
    }catch(const std::exception& er){
      cout << "[DTTTrigWriter] caught std::exception " << er.what() << endl;
    }catch(...){
      cout << "[DTTTrigWriter] Funny error" << endl;
    }
  }else{
    cout << "Service PoolDBOutputService is unavailable" << endl;
  }
}  

string DTTTrigWriter::getTBoxName(const DTSuperLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_hTimeBox";
  theStream >> histoName;
  return histoName;
}
