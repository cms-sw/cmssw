// -*- C++ -*-
//
// Package:    HcalDDDGeometryEP
// Class:      HcalDDDGeometryEP
// 
/**\class HcalDDDGeometryEP HcalDDDGeometryEP.h tmp/HcalDDDGeometryEP/interface/HcalDDDGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Oct 20 11:35:27 CDT 2006
// $Id: HcalDDDGeometryEP.cc,v 1.7 2012/10/29 07:32:07 mansj Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalDDDGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HcalDDDGeometryEP::HcalDDDGeometryEP(const edm::ParameterSet& ps ) :
   m_loader ( 0 ) ,
   m_cpv    ( 0 ) ,
   m_applyAlignment( ps.getUntrackedParameter<bool>("applyAlignment", false) ){

   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this,
		    &HcalDDDGeometryEP::produceAligned,
		    dependsOn( &HcalDDDGeometryEP::idealRecordCallBack ),
		    "HCAL");

// diable
//   setWhatProduced( this,
//		    &HcalDDDGeometryEP::produceIdeal,
//		    edm::es::Label( "HCAL" ) );
}


HcalDDDGeometryEP::~HcalDDDGeometryEP() { 
   delete m_loader ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceIdeal(const IdealGeometryRecord& iRecord) {
   idealRecordCallBack( iRecord ) ;

   edm::LogInfo("HCAL") << "Using default HCAL topology" ;
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;

   assert( 0 == m_loader ) ;
   m_loader = new HcalDDDGeometryLoader(*m_cpv); 
#ifdef DebugLog
   LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";
#endif
   ReturnType pC ( m_loader->load(*topology) ) ;

   return pC ;
}

HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceAligned(const HcalGeometryRecord& iRecord) {
   //now do what ever other initialization is needed
   assert( 0 != m_cpv ) ;

   edm::LogInfo("HCAL") << "Using default HCAL topology" ;
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;

   if( 0 == m_loader ) m_loader = new HcalDDDGeometryLoader(*m_cpv); 
#ifdef DebugLog
   LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";
#endif

   ReturnType pC ( m_loader->load(*topology) ) ;

   return pC ;
}

void
HcalDDDGeometryEP::idealRecordCallBack( const IdealGeometryRecord& iRecord ) {
   edm::ESTransientHandle<DDCompactView> pDD;
   iRecord.get( pDD );
   m_cpv = &(*pDD) ;
}


