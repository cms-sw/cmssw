/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/08/02 16:11:35 $
 *  $Revision: 1.5 $
 *  \author S. Bolognesi
 */

#include "CalibMuon/DTCalibration/plugins/DTTTrigWriter.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"



#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
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


// Constructor
DTTTrigWriter::DTTTrigWriter(const ParameterSet& pset) {
  // get selected debug option
  debug = pset.getUntrackedParameter<bool>("debug",false);

  // Open the root file which contains the histos
  theRootInputFile = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(theRootInputFile.c_str(), "READ");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);

  double sigmaFit = pset.getUntrackedParameter<double>("sigmaTTrigFit",10.);
  theFitter->setFitSigma(sigmaFit);

  // the kfactor to be uploaded in the ttrig DB
  kFactor = pset.getUntrackedParameter<double>("kFactor",-0.7);

  // Create the object to be written to DB
  tTrig = new DTTtrig();
  
  if(debug)
    cout << "[DTTTrigWriter]Constructor called!" << endl;
}



// Destructor
DTTTrigWriter::~DTTTrigWriter(){
  if(debug)
    cout << "[DTTTrigWriter]Destructor called!" << endl;
  theFile->Close();
  delete theFitter;
}



// Do the job
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
      
    // Get the histo from file
    DTSuperLayerId slId = (*sl)->id();
    TH1F* histo = (TH1F*)theFile->Get((getTBoxName(slId)).c_str());
    if(histo) { // Check that the histo exists
      // Compute mean and sigma of the rising edge
      pair<double, double> meanAndSigma = theFitter->fitTimeBox(histo);

      // Write them in DB object
      tTrig->set(slId,
		 meanAndSigma.first,
		 meanAndSigma.second,
                 kFactor,
		 DTTimeUnits::ns);
      if(debug) {
	cout << " SL: " << slId
	     << " mean = " << meanAndSigma.first
	     << " sigma = " << meanAndSigma.second << endl;
      }
    }
  }
}



// Write objects to DB
void DTTTrigWriter::endJob() {
  if(debug) 
	cout << "[DTTTrigWriter]Writing ttrig object to DB!" << endl;

  // FIXME: to be read from cfg?
  string tTrigRecord = "DTTtrigRcd";
  
  // Write the object to DB
  DTCalibDBUtils::writeToDB(tTrigRecord, tTrig);

}  



// Compute the name of the time box histo
string DTTTrigWriter::getTBoxName(const DTSuperLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_hTimeBox";
  theStream >> histoName;
  return histoName;
}
