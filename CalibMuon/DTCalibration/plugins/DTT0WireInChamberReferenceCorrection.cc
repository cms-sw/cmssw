/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/04/10 17:55:08 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "DTT0WireInChamberReferenceCorrection.h"
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

DTT0WireInChamberReferenceCorrection::DTT0WireInChamberReferenceCorrection(const ParameterSet& pset):
   calibChamber_( pset.getParameter<string>("calibChamber") ) {

   //DTChamberId chosenChamberId;
   if( calibChamber_ != "" && calibChamber_ != "None" && calibChamber_ != "All" ){
      stringstream linestr;
      int selWheel, selStation, selSector;
      linestr << calibChamber_;
      linestr >> selWheel >> selStation >> selSector;
      chosenChamberId_ = DTChamberId(selWheel, selStation, selSector);
      LogVerbatim("Calibration") << "[DTT0WireInChamberReferenceCorrection] Chosen chamber: " << chosenChamberId_ << endl;
   }
   //FIXME: Check if chosen chamber is valid.
}

DTT0WireInChamberReferenceCorrection::~DTT0WireInChamberReferenceCorrection() {
}

void DTT0WireInChamberReferenceCorrection::setES(const EventSetup& setup) {
   // Get t0 record from DB
   ESHandle<DTT0> t0H;
   setup.get<DTT0Rcd>().get(t0H);
   t0Map_ = &*t0H;
   LogVerbatim("Calibration") << "[DTT0WireInChamberReferenceCorrection] T0 version: " << t0H->version();

   // Get geometry from Event Setup
   setup.get<MuonGeometryRecord>().get(dtGeom_);
}

DTT0Data DTT0WireInChamberReferenceCorrection::correction(const DTWireId& wireId) {
   // Compute for selected chamber (or All) correction using as reference chamber mean

   DTChamberId chamberId = wireId.layerId().superlayerId().chamberId();

   if( calibChamber_ == "" || calibChamber_ == "None" )          return defaultT0(wireId);
   if( calibChamber_ != "All" && chamberId != chosenChamberId_ ) return defaultT0(wireId);

   // Access DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(status != 0) 
      throw cms::Exception("[DTT0WireInChamberReferenceCorrection]") << "Could not find t0 entry in DB for" 
                                                                     << wireId << endl;

   // Try to find t0 for reference wire in layer
   DTSuperLayerId slId = wireId.layerId().superlayerId();
   // Layers 1 and 2
   DTLayerId layerRef1( slId,1 );
   //DTLayerId layerRef2( slId,2 );

   const DTTopology& dtTopoLayerRef1 = dtGeom_->layer( layerRef1 )->specificTopology();
   const int firstWireLayerRef1 = dtTopoLayerRef1.firstChannel();
   const int refWireLayerRef1 = firstWireLayerRef1;
   DTWireId wireIdRefLayerRef1( layerRef1,refWireLayerRef1 );

   float t0MeanRef1,t0RMSRef1;
   int statusRef1 = t0Map_->get(wireIdRefLayerRef1,t0MeanRef1,t0RMSRef1,DTTimeUnits::counts);

   // Correct channels in a superlayer wrt t0 from first wire in first layer
   if(!statusRef1){
      float t0MeanNew = t0Mean - t0MeanRef1;
      float t0RMSNew = t0RMS;
      return DTT0Data(t0MeanNew,t0RMSNew);
   } else{
      // If reference wire not active could choose adjacent wire instead
      //...
      throw cms::Exception("[DTT0WireInChamberReferenceCorrection]") << "Could not find t0 entry in DB for" 
                                                                     << wireIdRefLayerRef1 << endl;
   }
}

DTT0Data DTT0WireInChamberReferenceCorrection::defaultT0(const DTWireId& wireId) {
   // Access default DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(!status){
      return DTT0Data(t0Mean,t0RMS);
   } else{
      //... 
      throw cms::Exception("[DTT0WireInChamberReferenceCorrection]") << "Could not find t0 entry in DB for"
	 << wireId << endl;
   }
}

} // namespace
