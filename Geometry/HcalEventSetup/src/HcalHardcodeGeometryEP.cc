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
// $Id: HcalHardcodeGeometryEP.cc,v 1.12 2012/03/22 10:30:51 sunanda Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalHardcodeGeometryEP.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
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


HcalHardcodeGeometryEP::~HcalHardcodeGeometryEP()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void
HcalHardcodeGeometryEP::idealRecordCallBack( const IdealGeometryRecord& iRecord )
{
}

HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produceIdeal( const IdealGeometryRecord& iRecord )
{

   edm::LogInfo("HCAL") << "Using default HCAL topology" ;
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;
   HcalFlexiHardcodeGeometryLoader loader(ps0);
   return ReturnType (loader.load (*topology));
}

HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produceAligned( const HcalGeometryRecord& iRecord )
{
  const IdealGeometryRecord& idealRecord = iRecord.getRecord<IdealGeometryRecord>();
  return produceIdeal (idealRecord);
}

