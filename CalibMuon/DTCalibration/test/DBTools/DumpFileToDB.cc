
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/05 09:14:26 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DumpFileToDB.h"
#include "DTCalibrationMap.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

using namespace edm;
using namespace std;

DumpFileToDB::DumpFileToDB(const ParameterSet& pset) {
  theCalibFile = new DTCalibrationMap(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));

  dbToDump = pset.getUntrackedParameter<string>("dbToDump", "TTrigDB");

  if(dbToDump != "TTrigDB" && dbToDump != "TZeroDB" && dbToDump != "NoiseDB")
    cout << "[DumpFileToDB] *** Error: parameter dbToDump is not valid, check the cfg file" << endl;
}

DumpFileToDB::~DumpFileToDB(){}


void DumpFileToDB::endJob() {
  if(dbToDump == "TTrigDB") { // Write the TTrig

    // Create the object to be written to DB
    DTTtrig* tTrig = new DTTtrig();

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      cout << "key: " << (*keyAndCalibs).first
	   << " ttrig_mean (ns): " << theCalibFile->tTrig((*keyAndCalibs).first)
	   << " ttrig_sigma(ns): " << theCalibFile->sigma_tTrig((*keyAndCalibs).first) << endl;
      tTrig->setSLTtrig((*keyAndCalibs).first.superlayerId(),
			theCalibFile->tTrig((*keyAndCalibs).first), 
			theCalibFile->sigma_tTrig((*keyAndCalibs).first),
			DTTimeUnits::ns);
    }

    cout << "[DumpFileToDB]Writing ttrig object to DB!" << endl;
    string record = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrig);

  } else if(dbToDump == "TZeroDB") { // Write the T0

    // Create the object to be written to DB
    DTT0* tZeroMap = new DTT0();

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      float t0mean = (*keyAndCalibs).second[4];
      float t0rms = (*keyAndCalibs).second[5];
      cout << "key: " << (*keyAndCalibs).first
	   << " T0 mean (TDC counts): " << t0mean
	   << " T0_rms (TDC counts): " << t0rms << endl;
      tZeroMap->setCellT0((*keyAndCalibs).first,
			  t0mean,
			  t0rms,
			  DTTimeUnits::counts );
    }

    cout << "[DumpFileToDB]Writing tZero object to DB!" << endl;
    string record = "DTT0Rcd";
    DTCalibDBUtils::writeToDB<DTT0>(record, tZeroMap);

  } else if(dbToDump == "NoiseDB") { // Write the Noise
    DTStatusFlag *statusMap = new DTStatusFlag();
    
    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      cout << "key: " << (*keyAndCalibs).first
	   << " Noisy flag: " << (*keyAndCalibs).second[6] << endl;
      statusMap->setCellNoise((*keyAndCalibs).first,
			      (*keyAndCalibs).second[6]);
    }

    cout << "[DumpFileToDB]Writing Noise Map object to DB!" << endl;
    string record = "DTStatusFlagRcd";
    DTCalibDBUtils::writeToDB<DTStatusFlag>(record, statusMap);
  }
}



