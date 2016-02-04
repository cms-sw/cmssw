
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/19 09:51:31 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "ProduceFakeDB.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

ProduceFakeDB::ProduceFakeDB(const ParameterSet& pset) {
  dbToProduce = pset.getUntrackedParameter<string>("dbToProduce", "TTrigDB");

  if(dbToProduce != "VDriftDB" && dbToProduce != "TTrigDB" && dbToProduce != "TZeroDB") 
    cout << "[ProduceFakeDB] *** Error: parameter dbToProduce is not valid, check the cfg file" << endl;

  ps = pset;
}

ProduceFakeDB::~ProduceFakeDB(){}

void ProduceFakeDB::beginRun(const edm::Run&, const EventSetup& setup) {
  setup.get<MuonGeometryRecord>().get(muonGeom);
}

void ProduceFakeDB::endJob() {
  
  //Get the superlayers and layers list
  vector<DTSuperLayer*> dtSupLylist = muonGeom->superLayers();
  vector<DTLayer*> dtLyList = muonGeom->layers();

  if(dbToProduce == "VDriftDB") {
    //Read the fake values from the cfg files
    double vdrift = ps.getUntrackedParameter<double>("vdrift", 0.00543);
    double hitReso = ps.getUntrackedParameter<double>("hitResolution", 0.02);

    // Create the object to be written to DB
    DTMtime* mtimeMap = new DTMtime();

    //Loop on superlayers
    for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
	 sl != dtSupLylist.end(); sl++) {
      // vdrift is cm/ns , resolution is cm
      mtimeMap->set((*sl)->id(), vdrift, hitReso, DTVelocityUnits::cm_per_ns);
    }

    // Write the object in the DB
    cout << "[ProduceFakeDB]Writing mtime object to DB!" << endl;
    string record = "DTMtimeRcd";
    DTCalibDBUtils::writeToDB<DTMtime>(record, mtimeMap);
  } 

  else if(dbToProduce == "TTrigDB") {
    //Read the fake values from the cfg files
    double ttrig = ps.getUntrackedParameter<double>("ttrig", 496);
    double sigmaTtrig = ps.getUntrackedParameter<double>("sigmaTtrig", 0);
    double kFactor = ps.getUntrackedParameter<double>("kFactor", 0);
    // Create the object to be written to DB
    DTTtrig* tTrigMap = new DTTtrig();

    //Loop on superlayers
    for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
	 sl != dtSupLylist.end(); sl++) {
      tTrigMap->set((*sl)->id(), ttrig, sigmaTtrig, kFactor, DTTimeUnits::ns);
    }

    // Write the object in the DB
    cout << "[ProduceFakeDB]Writing ttrig object to DB!" << endl;
    string record = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrigMap);
 }

  else if(dbToProduce == "TZeroDB") {
    //Read the fake values from the cfg files
    double tzero = ps.getUntrackedParameter<double>("tzero", 0);
    double sigmaTzero = ps.getUntrackedParameter<double>("sigmaTzero", 0);

    // Create the object to be written to DB
    DTT0* tZeroMap = new DTT0();

    //Loop on layers
    for (vector<DTLayer*>::const_iterator ly = dtLyList.begin();
	 ly != dtLyList.end(); ly++) {
	
      //Get the number of wires for each layer
      int nWires = (*ly)->specificTopology().channels();
      int firstWire = (*ly)->specificTopology().firstChannel();
      //Loop on wires
      for(int w=0; w<nWires; w++){
	DTWireId wireId((*ly)->id(), w + firstWire);
	tZeroMap->set(wireId, tzero, sigmaTzero, DTTimeUnits::counts );
      }
    }

    // Write the object in the DB
    cout << "[ProduceFakeDB]Writing tZero object to DB!" << endl;
    string record = "DTT0Rcd";
    DTCalibDBUtils::writeToDB<DTT0>(record, tZeroMap);

  }
}


