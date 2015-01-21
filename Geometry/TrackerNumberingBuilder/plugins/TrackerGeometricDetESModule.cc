#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CondDBCmsTrackerConstruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

using namespace edm;

TrackerGeometricDetESModule::TrackerGeometricDetESModule( const edm::ParameterSet & p ) 
  : fromDDD_( p.getParameter<bool>( "fromDDD" ))
{
  setWhatProduced( this );
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule( void ) {}

void
TrackerGeometricDetESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription descDB;
  descDB.add<bool>( "fromDDD", false );
  descriptions.add( "trackerNumberingGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<bool>( "fromDDD", true );
  descriptions.add( "trackerNumberingGeometry", desc );
}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );

    DDDCmsTrackerContruction theDDDCmsTrackerContruction;
    return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>( theDDDCmsTrackerContruction.construct(&(*cpv), dbl_to_int( DDVectorGetter::get( "detIdShifts" )))));
  }
  else
  {
    edm::ESHandle<PGeometricDet> pgd;
    iRecord.get( pgd );
    
    CondDBCmsTrackerConstruction cdbtc;
    return std::auto_ptr<GeometricDet> ( const_cast<GeometricDet*>( cdbtc.construct( *pgd )));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE( TrackerGeometricDetESModule );
