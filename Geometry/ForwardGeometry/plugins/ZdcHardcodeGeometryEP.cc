// -*- C++ -*-
//
// Package:    ZdcHardcodeGeometryEP
// Class:      ZdcHardcodeGeometryEP
// 
/**\class ZdcHardcodeGeometryEP ZdcHardcodeGeometryEP.h
   
    Description: <one line class summary>

    Implementation:
    <Notes on implementation>
*/
//
// Original Author:  Edmundo Garcia
//         Created:  Mon Aug  6 12:33:33 CDT 2007
// $Id: ZdcHardcodeGeometryEP.cc,v 1.1 2007/08/28 18:10:10 sunanda Exp $
//
#include "Geometry/ForwardGeometry/plugins/ZdcHardcodeGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


ZdcHardcodeGeometryEP::ZdcHardcodeGeometryEP( const edm::ParameterSet& ps ) :
   m_loader   (0),
   m_topology () ,
   m_applyAlignment ( ps.getUntrackedParameter<bool>("applyAlignment", false) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this,
		    &ZdcHardcodeGeometryEP::produceAligned,
		    dependsOn( &ZdcHardcodeGeometryEP::idealRecordCallBack ),
		    "ZDC");

   setWhatProduced( this,
		    &ZdcHardcodeGeometryEP::produceIdeal,
		    edm::es::Label( "ZDC" ) );
}


ZdcHardcodeGeometryEP::~ZdcHardcodeGeometryEP()
{ 
   delete m_loader ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void
ZdcHardcodeGeometryEP::idealRecordCallBack( const IdealGeometryRecord& iRecord )
{
}

ZdcHardcodeGeometryEP::ReturnType
ZdcHardcodeGeometryEP::produceIdeal( const IdealGeometryRecord& iRecord )
{
   assert( !m_applyAlignment ) ;

   ZdcHardcodeGeometryLoader loader ( m_topology ) ;

   ReturnType ptr ( loader.load() ) ;
   return ptr ;
}

ZdcHardcodeGeometryEP::ReturnType
ZdcHardcodeGeometryEP::produceAligned( const ZDCGeometryRecord& iRecord )
{
//   ZdcHardcodeGeometryLoader loader ( m_topology ) ;
   m_loader = new ZdcHardcodeGeometryLoader( m_topology ) ;

   ReturnType ptr ( m_loader->load() ) ;

   return ptr ;
}


