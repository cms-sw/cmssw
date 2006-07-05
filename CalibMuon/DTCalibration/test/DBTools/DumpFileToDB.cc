
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/03 15:09:40 $
 *  $Revision: 1.3 $
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

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
  if(dbToDump == "TTrigDB") {
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

    // Write the ttrig object to DB
    edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
    size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    if( dbOutputSvc.isAvailable() ){
      try{
	dbOutputSvc->newValidityForNewPayload<DTTtrig>(tTrig, dbOutputSvc->endOfTime(), callbackToken);
      }catch(const cond::Exception& er){
	cout << er.what() << endl;
      }catch(const std::exception& er){
	cout << "[DumpFileToDB] caught std::exception " << er.what() << endl;
      }catch(...){
	cout << "[DumpFileToDB] Funny error" << endl;
      }
    }else{
      cout << "Service PoolDBOutputService is unavailable" << endl;
    }

  } else if(dbToDump == "TZeroDB") {

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

    // Write the ttrig object to DB
    edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
    size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    if( dbOutputSvc.isAvailable() ){
      try{
	dbOutputSvc->newValidityForNewPayload<DTT0>(tZeroMap, dbOutputSvc->endOfTime(), callbackToken);
      }catch(const cond::Exception& er){
	cout << er.what() << endl;
      }catch(const std::exception& er){
	cout << "[DumpFileToDB] caught std::exception " << er.what() << endl;
      }catch(...){
	cout << "[DumpFileToDB] Funny error" << endl;
      }
    }else{
      cout << "Service PoolDBOutputService is unavailable" << endl;
    }

  } else if(dbToDump == "NoiseDB") {
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

    // Write the ttrig object to DB
    edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
    size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    if( dbOutputSvc.isAvailable() ){
      try{
	dbOutputSvc->newValidityForNewPayload<DTStatusFlag>(statusMap,
							    dbOutputSvc->endOfTime(),
							    callbackToken);
      }catch(const cond::Exception& er){
	cout << er.what() << endl;
      }catch(const std::exception& er){
	cout << "[DumpFileToDB] caught std::exception " << er.what() << endl;
      }catch(...){
	cout << "[DumpFileToDB] Funny error" << endl;
      }
    }else{
      cout << "Service PoolDBOutputService is unavailable" << endl;
    }
  }
}



