
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/02/15 16:45:47 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */

#include "ShiftTTrigDB.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

ShiftTTrigDB::ShiftTTrigDB(const ParameterSet& pset) {
  //Read the ttrig shifts
  shifts = pset.getParameter<vector<double> >("shifts");
  //Read the chambers to be shifted
  vector<ParameterSet> parameters =  pset.getParameter <vector<ParameterSet> > ("chambers");
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");

  int counter=0;
  for(vector<ParameterSet>::iterator parameter = parameters.begin(); parameter != parameters.end(); ++parameter ) {
    vector<int> chAddress;
    chAddress.push_back(parameter->getParameter<int>("wheel"));
    chAddress.push_back(parameter->getParameter<int>("sector"));
    chAddress.push_back(parameter->getParameter<int>("station"));
    chambers.push_back(chAddress);
    //Map the ttrig shift with the chamber addresses
    mapShiftsByChamber[chAddress] = shifts[counter];
    counter++;
  }

  debug = pset.getUntrackedParameter<bool>("debug",false);
  if (chambers.size() != shifts.size()){
    cout<<"[ShiftTTrigDB]: Wrong configuration: number of chambers different from number of shifts!! Aborting."<<endl;
    abort();
  }
}

ShiftTTrigDB::~ShiftTTrigDB(){}

void ShiftTTrigDB::beginRun(const edm::Run&, const EventSetup& setup) {
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap = &*tTrig;
  cout << "[ShiftTTrigDB]: TTrig version: " << tTrig->version() << endl;

  setup.get<MuonGeometryRecord>().get(muonGeom);
}


void ShiftTTrigDB::endJob() {
  // Create the object to be written to DB
  DTTtrig* tTrigNewMap = new DTTtrig();  
  //Get the superlayers list
  vector<DTSuperLayer*> dtSupLylist = muonGeom->superLayers();

  //Loop on all superlayers
  for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
       sl != dtSupLylist.end(); sl++) {
    float ttrigMean = 0;
    float ttrigSigma = 0;
    float kFactor = 0;
    tTrigMap->get((*sl)->id(),
		  ttrigMean,
		  ttrigSigma,
                  kFactor,
		  DTTimeUnits::ns);
    bool ttrigShifted = false;
    //Loop on the chambers with ttrig to be shifted
    for(vector<vector<int> >::const_iterator ch =chambers.begin();
	ch != chambers.end(); ch++){
      //Check the chamber address format
      if((*ch).size()!=3){
	cout<<"[ShiftTTrigDB]: Wrong configuration: use three integer to indicate each chamber. Aborting."<<endl;
	abort();
      }
      if(((*sl)->id().wheel() == (*ch)[0]) ||  (*ch)[0] == 999){
	if(((*sl)->id().sector() == (*ch)[1]) ||  (*ch)[1] == 999){
	  if(((*sl)->id().station() == (*ch)[2]) ||  (*ch)[2] == 999){

	    //Compute new ttrig
	    double newTTrigMean =  ttrigMean + mapShiftsByChamber[(*ch)]; 
	    //Store new ttrig in the new map
	    tTrigNewMap->set((*sl)->id(), newTTrigMean, ttrigSigma, kFactor, DTTimeUnits::ns);
	    ttrigShifted = true;
	    if(debug){
	      cout<<"Shifting SL: " << (*sl)->id()
		  << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	    }
	  }
	}
      }
    }//End loop on chambers to be shifted
    if(!ttrigShifted){
      //Store old ttrig in the new map
      tTrigNewMap->set((*sl)->id(), ttrigMean, ttrigSigma, kFactor, DTTimeUnits::ns);
      if(debug){
	cout<<"Copying  SL: " << (*sl)->id()
	    << " ttrig "<< ttrigMean <<endl;
      }
    }
  }//End of loop on superlayers 

  //Write object to DB
 cout << "[ShiftTTrigDB]: Writing ttrig object to DB!" << endl;
    string record = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrigNewMap);
} 



