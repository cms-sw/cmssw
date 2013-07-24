/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/04/10 17:55:08 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "DTT0AbsoluteReferenceCorrection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTT0AbsoluteReferenceCorrection::DTT0AbsoluteReferenceCorrection(const ParameterSet& pset):
   calibChamber_( pset.getParameter<string>("calibChamber") ),
   reference_( pset.getParameter<double>("reference") ) {

   //DTChamberId chosenChamberId;
   if( calibChamber_ != "" && calibChamber_ != "None" && calibChamber_ != "All" ){
      stringstream linestr;
      int selWheel, selStation, selSector;
      linestr << calibChamber_;
      linestr >> selWheel >> selStation >> selSector;
      chosenChamberId_ = DTChamberId(selWheel, selStation, selSector);
      LogVerbatim("Calibration") << "[DTT0AbsoluteReferenceCorrection] Chosen chamber: " << chosenChamberId_ << endl;
   }
   //FIXME: Check if chosen chamber is valid.
}

DTT0AbsoluteReferenceCorrection::~DTT0AbsoluteReferenceCorrection() {
}

void DTT0AbsoluteReferenceCorrection::setES(const EventSetup& setup) {
   // Get t0 record from DB
   ESHandle<DTT0> t0H;
   setup.get<DTT0Rcd>().get(t0H);
   t0Map_ = &*t0H;
   LogVerbatim("Calibration") << "[DTT0AbsoluteReferenceCorrection] T0 version: " << t0H->version();
}

DTT0Data DTT0AbsoluteReferenceCorrection::correction(const DTWireId& wireId) {
   // Compute for selected chamber (or All) correction using as reference chamber mean

   DTChamberId chamberId = wireId.layerId().superlayerId().chamberId();

   if( calibChamber_ == "" || calibChamber_ == "None" )          return defaultT0(wireId);
   if( calibChamber_ != "All" && chamberId != chosenChamberId_ ) return defaultT0(wireId);

   // Access DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(status != 0) 
      throw cms::Exception("[DTT0AbsoluteReferenceCorrection]") << "Could not find t0 entry in DB for" 
                                                                << wireId << endl;

   float t0MeanNew = t0Mean - reference_;
   float t0RMSNew = t0RMS;
   return DTT0Data(t0MeanNew,t0RMSNew);
}

DTT0Data DTT0AbsoluteReferenceCorrection::defaultT0(const DTWireId& wireId) {
   // Access default DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(!status){
      return DTT0Data(t0Mean,t0RMS);
   } else{
      //... 
      throw cms::Exception("[DTT0AbsoluteReferenceCorrection]") << "Could not find t0 entry in DB for"
	 << wireId << endl;
   }
}

} // namespace
