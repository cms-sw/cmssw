#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CondDBCmsTrackerConstruction.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"
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
  auto cc = setWhatProduced( this );
  if(fromDDD_) {
    ddToken_ = cc.consumes<DDCompactView>(edm::ESInputTag());
  } else {
    pgToken_ = cc.consumes<PGeometricDet>(edm::ESInputTag());
  }
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

std::unique_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle( ddToken_ );

    return DDDCmsTrackerContruction::construct(*cpv, dbl_to_int( DDVectorGetter::get( "detIdShifts" )));
  }
  else
  {
    auto const& pgd = iRecord.get( pgToken_ );
    
    return CondDBCmsTrackerConstruction::construct( pgd );
  }
}

DEFINE_FWK_EVENTSETUP_MODULE( TrackerGeometricDetESModule );
