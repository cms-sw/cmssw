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
// $Id: HcalHardcodeGeometryEP.cc,v 1.7 2008/04/21 22:18:19 heltsley Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalHardcodeGeometryEP.h"
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

HcalHardcodeGeometryEP::HcalHardcodeGeometryEP( const edm::ParameterSet& ps ) :
   m_loader ( 0 ) ,
   m_applyAlignment ( ps.getUntrackedParameter<bool>("applyAlignment", false) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this,
		    &HcalHardcodeGeometryEP::produceAligned,
		    dependsOn( &HcalHardcodeGeometryEP::idealRecordCallBack ),
		    "HCAL");

// disable
//   setWhatProduced( this,
//		    &HcalHardcodeGeometryEP::produceIdeal,
//		    edm::es::Label( "HCAL" ) );
}


HcalHardcodeGeometryEP::~HcalHardcodeGeometryEP()
{ 
   delete m_loader ;
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
   assert( !m_applyAlignment ) ;

   //now do what ever other initialization is needed
   ReturnType ptr ;
   edm::LogInfo("HCAL") << "Using default HCAL topology" ;
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;
   m_loader = new HcalHardcodeGeometryLoader( *topology ) ;
   ptr = ReturnType( m_loader->load() ) ;
   return ptr ;
}

HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produceAligned( const HcalGeometryRecord& iRecord )
{
   //now do what ever other initialization is needed
   ReturnType ptr ;
   edm::LogInfo("HCAL") << "Using default HCAL topology" ;

   edm::ESHandle<HcalTopology> topology ;
   iRecord.getRecord<IdealGeometryRecord>().get( topology ) ;
   if( 0 == m_loader ) m_loader = new HcalHardcodeGeometryLoader( *topology ) ;
   ptr = ReturnType( m_loader->load() ) ; 
   return ptr ;
}
