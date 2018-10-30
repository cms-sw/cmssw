#include "Geometry/MTDNumberingBuilder/plugins/MTDGeometricTimingDetESModule.h"
#include "Geometry/MTDNumberingBuilder/plugins/DDDCmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/CondDBCmsMTDConstruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
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

MTDGeometricTimingDetESModule::MTDGeometricTimingDetESModule( const edm::ParameterSet & p ) 
  : fromDDD_( p.getParameter<bool>( "fromDDD" ))
{
  setWhatProduced( this );
}

MTDGeometricTimingDetESModule::~MTDGeometricTimingDetESModule( void ) {}

void
MTDGeometricTimingDetESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription descDB;
  descDB.add<bool>( "fromDDD", false );
  descriptions.add( "mtdNumberingGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<bool>( "fromDDD", true );
  descriptions.add( "mtdNumberingGeometry", desc );
}

std::unique_ptr<GeometricTimingDet> 
MTDGeometricTimingDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );

    DDDCmsMTDConstruction theDDDCmsMTDConstruction;
    return std::unique_ptr<GeometricTimingDet> (const_cast<GeometricTimingDet*>( theDDDCmsMTDConstruction.construct(&(*cpv), dbl_to_int( DDVectorGetter::get( "detIdShifts" )))));

  }
  else
  {
    edm::ESHandle<PGeometricTimingDet> pgd;
    iRecord.get( pgd );
    
    CondDBCmsMTDConstruction cdbtc;
    return std::unique_ptr<GeometricTimingDet> ( const_cast<GeometricTimingDet*>( cdbtc.construct( *pgd )));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE( MTDGeometricTimingDetESModule );
