
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/06/16 12:22:38 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DumpFileToDB.h"
#include "DTCalibrationFile.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

using namespace edm;
using namespace std;

DumpFileToDB::DumpFileToDB(const ParameterSet& pset) {
  theCalibFile = new DTCalibrationFile(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));
}

DumpFileToDB::~DumpFileToDB(){}


void DumpFileToDB::endJob() {
  // Create the object to be written to DB
  DTTtrig* tTrig = new DTTtrig();

  // Loop over file entries
  for(DTCalibrationFile::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
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
}





