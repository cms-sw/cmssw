#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CondDBCmsTrackerConstruction.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

using namespace edm;

TrackerGeometricDetESModule::TrackerGeometricDetESModule( const edm::ParameterSet & p ) 
  : fromDDD_( p.getParameter<bool>( "fromDDD" )),
    layerNumberPXB_( 16 ), // 18 for SLHC p.getParameter<unsigned int>( "layerNumberPXB" )),
    totalBlade_( 24 ) 	   // 56 for SLHC p.getParameter<unsigned int>( "totalBlade" ))
{
  const edm::ParameterSet tkGeomConsts( p.getParameter<edm::ParameterSet>( "pixelGeometryConstants" ));
  layerNumberPXB_ = tkGeomConsts.getParameter<unsigned int>( "layerNumberPXB" );
  totalBlade_ = tkGeomConsts.getParameter<unsigned int>( "totalBlade" );
  
  setWhatProduced( this );
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule( void ) {}

void
TrackerGeometricDetESModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription descPixelGeometryConstants;
  descPixelGeometryConstants.add<unsigned int>("layerNumberPXB", 16U);
  descPixelGeometryConstants.add<unsigned int>("totalBlade", 24U);

  edm::ParameterSetDescription descPixelSLHCGeometryConstants;
  descPixelSLHCGeometryConstants.add<unsigned int>("layerNumberPXB", 18U);
  descPixelSLHCGeometryConstants.add<unsigned int>("totalBlade", 56U);

  edm::ParameterSetDescription descDB;
  descDB.add<bool>("fromDDD", false);
  descDB.addOptional<edm::ParameterSetDescription>("pixelGeometryConstants", descPixelGeometryConstants);
  descriptions.add("trackerNumberingGeometryDB", descDB);

  edm::ParameterSetDescription descSLHCDB;
  descSLHCDB.add<bool>("fromDDD", false);
  descSLHCDB.addOptional<edm::ParameterSetDescription>("pixelGeometryConstants", descPixelSLHCGeometryConstants);
  descriptions.add("trackerNumberingSLHCGeometryDB", descSLHCDB);

  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.addOptional<edm::ParameterSetDescription>("pixelGeometryConstants", descPixelGeometryConstants);
  descriptions.add("trackerNumberingGeometry", desc);

  edm::ParameterSetDescription descSLHC;
  descSLHC.add<bool>("fromDDD", true);
  descSLHC.addOptional<edm::ParameterSetDescription>("pixelGeometryConstants", descPixelSLHCGeometryConstants);
  descriptions.add("trackerNumberingSLHCGeometryDB", descSLHC);
}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    
    DDDCmsTrackerContruction theDDDCmsTrackerContruction;
    return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>( theDDDCmsTrackerContruction.construct(&(*cpv), layerNumberPXB_, totalBlade_ )));
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
