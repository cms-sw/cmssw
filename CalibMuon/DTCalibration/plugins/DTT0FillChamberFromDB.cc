/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/21 13:48:46 $
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "DTT0FillChamberFromDB.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTT0FillChamberFromDB::DTT0FillChamberFromDB(const ParameterSet& pset):
  dbLabelRef_( pset.getParameter<string>("dbLabelRef") ),
  chamberRef_( pset.getParameter<string>("chamberId") ) {

  //DTChamberId chosenChamberId;
  if( chamberRef_ != "" && chamberRef_ != "None" ){
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << chamberRef_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId_ = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << "[DTT0FillChamberFromDB] Chosen chamber: " << chosenChamberId_ << endl;
  }
  //FIXME: Check if chosen chamber is valid.

}

DTT0FillChamberFromDB::~DTT0FillChamberFromDB() {
}

void DTT0FillChamberFromDB::setES(const EventSetup& setup) {
  // Get t0 record from DB
  ESHandle<DTT0> t0H;
  setup.get<DTT0Rcd>().get(t0H);
  t0Map_ = &*t0H;
  LogVerbatim("Calibration") << "[DTT0FillChamberFromDB] T0 version: " << t0H->version();

  // Get reference t0 DB 
  ESHandle<DTT0> t0RefH;
  setup.get<DTT0Rcd>().get(dbLabelRef_,t0RefH);
  t0MapRef_ = &*t0RefH;
  LogVerbatim("Calibration") << "[DTT0FillChamberFromDB] Reference T0 version: " << t0RefH->version();

}

DTT0Data DTT0FillChamberFromDB::correction(const DTWireId& wireId) {
  // If wire belongs to chosen chamber, use t0 value from reference DB
  // Otherwise use value from default DB
 
  DTChamberId chamberId = wireId.layerId().superlayerId().chamberId();

  if( chamberRef_ != "" && chamberRef_ != "None" && chamberId == chosenChamberId_ ){
     // Access reference DB
     float t0MeanRef,t0RMSRef;
     int statusRef = t0MapRef_->get(wireId,t0MeanRef,t0RMSRef,DTTimeUnits::counts);
     if(!statusRef){
        return DTT0Data(t0MeanRef,t0RMSRef);
     } else{
        //... 
        throw cms::Exception("[DTT0FillChamberFromDB]") << "Could not find t0 entry in reference DB for"
                                                        << wireId << endl;
     }
  } else{
     // Access default DB
     float t0Mean,t0RMS;
     int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
     if(!status){
	return DTT0Data(t0Mean,t0RMS);
     } else{
        //... 
        throw cms::Exception("[DTT0FillChamberFromDB]") << "Could not find t0 entry in DB for"
                                                        << wireId << endl;
     }
  }
}

} // namespace
