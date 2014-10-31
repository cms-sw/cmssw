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
    detidShifts_( p.exists( "detidShifts" ) ? p.getParameter<std::vector<int> >( "detidShifts" ) : std::vector<int>() )
{
  // if the vector of detid shifts in not configured, fill it with the default values
  if(detidShifts_.size()==0) {
    // level 0
    detidShifts_.push_back(-1); detidShifts_.push_back(23); detidShifts_.push_back(-1); 
    detidShifts_.push_back(13); detidShifts_.push_back(-1); detidShifts_.push_back(18);
    // level 1
    detidShifts_.push_back(16); detidShifts_.push_back(16); detidShifts_.push_back(14); 
    detidShifts_.push_back(11); detidShifts_.push_back(14); detidShifts_.push_back(14);
    // level 2
    detidShifts_.push_back(8) ; detidShifts_.push_back(8) ; detidShifts_.push_back(4); 
    detidShifts_.push_back(9) ; detidShifts_.push_back(5) ; detidShifts_.push_back(8);
    // level 3
    detidShifts_.push_back(2) ; detidShifts_.push_back(2) ; detidShifts_.push_back(2); 
    detidShifts_.push_back(2) ; detidShifts_.push_back(2) ; detidShifts_.push_back(5);
    // level 4
    detidShifts_.push_back(0) ; detidShifts_.push_back(0) ; detidShifts_.push_back(0); 
    detidShifts_.push_back(0) ; detidShifts_.push_back(0) ; detidShifts_.push_back(2);
    // level 5
    detidShifts_.push_back(-1); detidShifts_.push_back(-1); detidShifts_.push_back(-1); 
    detidShifts_.push_back(-1); detidShifts_.push_back(-1); detidShifts_.push_back(0);
  }

  setWhatProduced( this );
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule( void ) {}

void
TrackerGeometricDetESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  std::vector<int> presentDet;
    // level 0
  presentDet.push_back(-1); presentDet.push_back(23); presentDet.push_back(-1); 
  presentDet.push_back(13); presentDet.push_back(-1); presentDet.push_back(18);
  // level 1
  presentDet.push_back(16); presentDet.push_back(16); presentDet.push_back(14); 
  presentDet.push_back(11); presentDet.push_back(14); presentDet.push_back(14);
  // level 2
  presentDet.push_back(8) ; presentDet.push_back(8) ; presentDet.push_back(4); 
  presentDet.push_back(9) ; presentDet.push_back(5) ; presentDet.push_back(8);
  // level 3
  presentDet.push_back(2) ; presentDet.push_back(2) ; presentDet.push_back(2); 
  presentDet.push_back(2) ; presentDet.push_back(2) ; presentDet.push_back(5);
  // level 4
  presentDet.push_back(0) ; presentDet.push_back(0) ; presentDet.push_back(0); 
  presentDet.push_back(0) ; presentDet.push_back(0) ; presentDet.push_back(2);
  // level 5
  presentDet.push_back(-1); presentDet.push_back(-1); presentDet.push_back(-1); 
  presentDet.push_back(-1); presentDet.push_back(-1); presentDet.push_back(0);

  std::vector<int> slhcDet;
    // level 0
  slhcDet.push_back(-1); slhcDet.push_back(23); slhcDet.push_back(-1); 
  slhcDet.push_back(13); slhcDet.push_back(-1); slhcDet.push_back(18);
  // level 1
  slhcDet.push_back(20); slhcDet.push_back(18); slhcDet.push_back(14); 
  slhcDet.push_back(11); slhcDet.push_back(14); slhcDet.push_back(14);
  // level 2
  slhcDet.push_back(12); slhcDet.push_back(10); slhcDet.push_back(4); 
  slhcDet.push_back(9) ; slhcDet.push_back(5) ; slhcDet.push_back(8);
  // level 3
  slhcDet.push_back(2) ; slhcDet.push_back(2) ; slhcDet.push_back(2); 
  slhcDet.push_back(2) ; slhcDet.push_back(2) ; slhcDet.push_back(5);
  // level 4
  slhcDet.push_back(0) ; slhcDet.push_back(0) ; slhcDet.push_back(0); 
  slhcDet.push_back(0) ; slhcDet.push_back(0) ; slhcDet.push_back(2);
  // level 5
  slhcDet.push_back(-1); slhcDet.push_back(-1); slhcDet.push_back(-1); 
  slhcDet.push_back(-1); slhcDet.push_back(-1); slhcDet.push_back(0);
  
  edm::ParameterSetDescription descDB;
  descDB.add<bool>( "fromDDD", false );
  descDB.addOptional<std::vector<int> >( "detidShifts", presentDet );
  descriptions.add( "trackerNumberingGeometryDB", descDB );

  edm::ParameterSetDescription descSLHCDB;
  descSLHCDB.add<bool>( "fromDDD", false );
  descSLHCDB.addOptional<std::vector<int> >( "detidShifts", slhcDet );
  descriptions.add( "trackerNumberingSLHCGeometryDB", descSLHCDB );

  edm::ParameterSetDescription desc;
  desc.add<bool>( "fromDDD", true );
  desc.addOptional<std::vector<int> >( "detidShifts", presentDet );
  descriptions.add( "trackerNumberingGeometry", desc );

  edm::ParameterSetDescription descSLHC;
  descSLHC.add<bool>( "fromDDD", true );
  descSLHC.addOptional<std::vector<int> >( "detidShifts", slhcDet );
  descriptions.add( "trackerNumberingSLHCGeometry", descSLHC );
}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    
    DDDCmsTrackerContruction theDDDCmsTrackerContruction;
    return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>( theDDDCmsTrackerContruction.construct(&(*cpv), detidShifts_)));
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
