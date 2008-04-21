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
// $Id: HcalDDDGeometryEP.cc,v 1.2 2008/01/22 21:35:42 muzaffar Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalDDDGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
   m_applyAlignment ( ps.getUntrackedParameter<bool>("applyAlignment", false) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this,
		    &HcalDDDGeometryEP::produceAligned,
		    dependsOn( &HcalDDDGeometryEP::idealRecordCallBack ),
		    "HCAL");

   setWhatProduced( this,
		    &HcalDDDGeometryEP::produceIdeal,
		    edm::es::Label( "HCAL" ) );
}


HcalDDDGeometryEP::~HcalDDDGeometryEP() 
{ 
   delete m_loader ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceIdeal(const IdealGeometryRecord& iRecord)
{
   idealRecordCallBack( iRecord ) ;

   assert( 0 == m_loader ) ;
   m_loader = new HcalDDDGeometryLoader(*m_cpv); 
   LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";

   ReturnType pC ( m_loader->load() ) ;

   return pC ;
}
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceAligned(const HcalGeometryRecord& iRecord)
{
   //now do what ever other initialization is needed
   assert( 0 != m_cpv ) ;
   if( 0 == m_loader ) m_loader = new HcalDDDGeometryLoader(*m_cpv); 
   LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";

   ReturnType pC ( m_loader->load() ) ;

   return pC ;
}

void
HcalDDDGeometryEP::idealRecordCallBack( const IdealGeometryRecord& iRecord )
{
   edm::ESHandle< DDCompactView > pDD;
   iRecord.get( pDD );
   m_cpv = &(*pDD) ;
}


