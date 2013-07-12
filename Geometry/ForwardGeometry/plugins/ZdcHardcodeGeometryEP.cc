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
// $Id: ZdcHardcodeGeometryEP.cc,v 1.3 2008/05/19 20:12:41 heltsley Exp $
//
#include "Geometry/Records/interface/ZDCGeometryRecord.h"
#include "Geometry/ForwardGeometry/plugins/ZdcHardcodeGeometryEP.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


ZdcHardcodeGeometryEP::ZdcHardcodeGeometryEP( const edm::ParameterSet& ps ) :
   m_loader   (0),
   m_topology () ,
   m_applyAlignment ( ps.getUntrackedParameter<bool>("applyAlignment", false) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this,
		    ZdcGeometry::producerTag() );

// disable
//   setWhatProduced( this,
//		    &ZdcHardcodeGeometryEP::produceIdeal,
//		    edm::es::Label( "ZDC" ) );
}


ZdcHardcodeGeometryEP::~ZdcHardcodeGeometryEP()
{ 
   delete m_loader ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------

ZdcHardcodeGeometryEP::ReturnType
ZdcHardcodeGeometryEP::produce( const ZDCGeometryRecord& iRecord )
{
//   ZdcHardcodeGeometryLoader loader ( m_topology ) ;
   m_loader = new ZdcHardcodeGeometryLoader( m_topology ) ;

   ReturnType ptr ( m_loader->load() ) ;

   return ptr ;
}


