
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/08 10:17:18 $
 *  $Revision: 1.6 $
 *  \author M. Giunta
 */

#include "CalibMuon/DTCalibration/plugins/DTVDriftWriter.h"
#include "CalibMuon/DTCalibration/interface/DTMeanTimerFitter.h"
#include "CalibMuon/DTCalibration/interface/vDriftHistos.h"
#include "CalibMuon/DTCalibration/plugins/DTCalibrationMap.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"

/* C++ Headers */
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TFile.h"

using namespace std;
using namespace edm;
//using namespace dttmaxenums;



DTVDriftWriter::DTVDriftWriter(const ParameterSet& pset) {
  // get selected debug option
  debug = pset.getUntrackedParameter<bool>("debug", "false");

  // Open the root file which contains the histos
  theRootInputFile = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(theRootInputFile.c_str(), "READ");
  
  theFitter = new DTMeanTimerFitter(theFile);
  if(debug)
    theFitter->setVerbosity(1);

  // the text file which will contain the histos
  theVDriftOutputFile = pset.getUntrackedParameter<string>("vDriftFileName");

  // get parameter set for DTCalibrationMap constructor
  theCalibFilePar =  pset.getUntrackedParameter<ParameterSet>("calibFileConfig");

  // the granularity to be used for calib constants evaluation
  theGranularity = pset.getUntrackedParameter<string>("calibGranularity","bySL");
  
  theMTime = new DTMtime();

  if(debug)
    cout << "[DTVDriftWriter]Constructor called!" << endl;
}


DTVDriftWriter::~DTVDriftWriter(){
  if(debug)
    cout << "[DTVDriftWriter]Destructor called!" << endl;
  theFile->Close();
  delete theFitter;
}

void DTVDriftWriter::analyze(const Event & event, const EventSetup& eventSetup) {
  if(debug)
    cout << "[DTVDriftWriter]Analyzer called!" << endl;

  // Instantiate a DTCalibrationMap object 
  DTCalibrationMap calibValuesFile(theCalibFilePar);  

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  if(theGranularity == "bySL") {    
    // Get all the sls from the setup
    const vector<DTSuperLayer*> superLayers = dtGeom->superLayers(); 
    
    // Loop over all SLs
    for(vector<DTSuperLayer*>::const_iterator  slCell = superLayers.begin();
	slCell != superLayers.end(); slCell++) {
      
      DTSuperLayerId slId = (*slCell)->id();
      // evaluate v_drift and sigma from the TMax histograms
      DTWireId wireId(slId, 0, 0);
      vector<float> newConstants;
      TString N=(((((TString) "TMax"+(long) wireId.wheel()) +(long) wireId.station())
		  +(long) wireId.sector())+(long) wireId.superLayer());
      vector<float> vDriftAndReso = theFitter->evaluateVDriftAndReso(N);

      // Don't write the constants for the SL if the vdrift was not computed
      if(vDriftAndReso.front() == -1)
	continue;

      const DTCalibrationMap::CalibConsts* oldConstants = calibValuesFile.getConsts(wireId);
      
      if(oldConstants != 0) {
	newConstants.push_back((*oldConstants)[0]);
	newConstants.push_back((*oldConstants)[1]);
	newConstants.push_back((*oldConstants)[2]);
      } else {
	newConstants.push_back(-1);
	newConstants.push_back(-1);
	newConstants.push_back(-1);
      }
      for(int ivd=0; ivd<=5;ivd++) { 
	// 0=vdrift, 1=reso, 2=(3deltat0-2deltat0), 3=(2deltat0-1deltat0),
	//  4=(1deltat0-0deltat0), 5=deltat0 from hists with max entries,
	newConstants.push_back(vDriftAndReso[ivd]); 
      }
      calibValuesFile.addCell(calibValuesFile.getKey(wireId), newConstants);

      // vdrift is cm/ns , resolution is cm
      theMTime->set(slId,
		    vDriftAndReso[0],
		    vDriftAndReso[1],
		    DTVelocityUnits::cm_per_ns);
      if(debug) {
	cout << " SL: " << slId
	     << " vDrift = " << vDriftAndReso[0]
	     << " reso = " << vDriftAndReso[1] << endl;
      }
    }
  }
  // to be implemented: granularity different from bySL

  //   if(theGranularity == "byChamber") {
  //     const vector<DTChamber*> chambers = dMap.chambers();
    
  //     // Loop over all chambers
  //     for(vector<MuBarChamber*>::const_iterator chamber = chambers.begin();
  // 	chamber != chambers.end(); chamber ++) {
  //       MuBarChamberId chamber_id = (*chamber)->id();
  //       MuBarDigiParameters::Key wire_id(chamber_id, 0, 0, 0);
  //       vector<float> newConstants;
  //       vector<float> vDriftAndReso = evaluateVDriftAndReso(wire_id, f);
  //       const CalibConsts* oldConstants = digiParams.getConsts(wire_id);
  //       if(oldConstants !=0) {
  // 	newConstants = *oldConstants;
  // 	newConstants.push_back(vDriftAndReso[0]);
  // 	newConstants.push_back(vDriftAndReso[1]);
  // 	newConstants.push_back(vDriftAndReso[2]);
  // 	newConstants.push_back(vDriftAndReso[3]);
  //       } else {
  // 	newConstants.push_back(-1);
  // 	newConstants.push_back(-1);
  // 	newConstants.push_back(vDriftAndReso[0]);
  // 	newConstants.push_back(vDriftAndReso[1]);
  // 	newConstants.push_back(vDriftAndReso[2]);
  // 	newConstants.push_back(vDriftAndReso[3]);
  //       }
  //       digiParams.addCell(wire_id, newConstants);
  //     }
  //   }
  //write values to a table  
  calibValuesFile.writeConsts(theVDriftOutputFile);
}



void DTVDriftWriter::endJob() {

  if(debug) 
    cout << "[DTVDriftWriter]Writing vdrift object to DB!" << endl;

  // Write the MeanTimer object to DB
  string record = "DTMtimeRcd";
  DTCalibDBUtils::writeToDB<DTMtime>(record, theMTime);
}



  // to be implemented: granularity different from bySL

  // // Create partitions 
  // DTVDriftWriter::cellInfo* DTVDriftWriter::partition(const DTWireId& wireId) {
  //   for( map<MuBarWireId, cellInfo*>::const_iterator iter =
  // 	 mapCellTmaxPart.begin(); iter != mapCellTmaxPart.end(); iter++) {
  //     // Divide wires per SL (with phi symmetry)
  //     if(iter->first.wheel() == wireId.wheel() &&
  //        iter->first.station() == wireId.station() &&
  //        //       iter->first.sector() == wireId.sector() && // phi symmetry!
  //        iter->first.superlayer() == wireId.superlayer()) {
  //       return iter->second;
  //     }
  //   }
  //   cellInfo * result = new cellInfo("dummy string"); // FIXME: change constructor; create tree?
  //   mapCellTmaxPart.insert(make_pair(wireId, result));
  //   return result;
  //}
