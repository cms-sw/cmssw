// -*- C++ -*-
//
// Package:    HcalHardcodeGeometryEP
// Class:      HcalHardcodeGeometryEP
// 
/**\class HcalHardcodeGeometryEP HcalHardcodeGeometryEP.h tmp/HcalHardcodeGeometryEP/interface/HcalHardcodeGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/HcalEventSetup/interface/HcalHardcodeGeometryEP.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HcalTopology;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HcalHardcodeGeometryEP::HcalHardcodeGeometryEP( const edm::ParameterSet& ps ) : ps0(ps) {
  
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced( this,
		   &HcalHardcodeGeometryEP::produceAligned,
		   dependsOn( &HcalHardcodeGeometryEP::idealRecordCallBack ),
		   HcalGeometry::producerTag() );

// disable
//   setWhatProduced( this,
//		    &HcalHardcodeGeometryEP::produceIdeal,
//		    edm::es::Label( "HCAL" ) );
}


HcalHardcodeGeometryEP::~HcalHardcodeGeometryEP() { }


//
// member functions
//


HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produceIdeal( const HcalRecNumberingRecord& iRecord ) {

   edm::LogInfo("HCAL") << "Using default HCAL topology" ;
   edm::ESHandle<HcalDDDRecConstants> hcons;
   iRecord.get( hcons ) ;
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;
   HcalFlexiHardcodeGeometryLoader loader(ps0);
   return ReturnType (loader.load (*topology, *hcons));
}

HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produceAligned( const HcalGeometryRecord& iRecord ) {
  const HcalRecNumberingRecord& idealRecord = iRecord.getRecord<HcalRecNumberingRecord>();
  return produceIdeal (idealRecord);
}

