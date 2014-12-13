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

  std::vector<int> phase1Det;
    // level 0
  phase1Det.push_back(-1); phase1Det.push_back(23); phase1Det.push_back(-1); 
  phase1Det.push_back(13); phase1Det.push_back(-1); phase1Det.push_back(18);
  // level 1
  phase1Det.push_back(20); phase1Det.push_back(18); phase1Det.push_back(14); 
  phase1Det.push_back(11); phase1Det.push_back(14); phase1Det.push_back(14);
  // level 2
  phase1Det.push_back(12); phase1Det.push_back(10); phase1Det.push_back(4); 
  phase1Det.push_back(9) ; phase1Det.push_back(5) ; phase1Det.push_back(8);
  // level 3
  phase1Det.push_back(2) ; phase1Det.push_back(2) ; phase1Det.push_back(2); 
  phase1Det.push_back(2) ; phase1Det.push_back(2) ; phase1Det.push_back(5);
  // level 4
  phase1Det.push_back(0) ; phase1Det.push_back(0) ; phase1Det.push_back(0); 
  phase1Det.push_back(0) ; phase1Det.push_back(0) ; phase1Det.push_back(2);
  // level 5
  phase1Det.push_back(-1); phase1Det.push_back(-1); phase1Det.push_back(-1); 
  phase1Det.push_back(-1); phase1Det.push_back(-1); phase1Det.push_back(0);

  std::vector<int> phase2Det;
    // level 0
  phase2Det.push_back(-1); phase2Det.push_back(23); phase2Det.push_back(-1); 
  phase2Det.push_back(23); phase2Det.push_back(-1); phase2Det.push_back(18);
  // level 1
  phase2Det.push_back(20); phase2Det.push_back(18); phase2Det.push_back(14); 
  phase2Det.push_back(18); phase2Det.push_back(20); phase2Det.push_back(14);
  // level 2
  phase2Det.push_back(12); phase2Det.push_back(10); phase2Det.push_back(4); 
  phase2Det.push_back(10) ; phase2Det.push_back(12); phase2Det.push_back(8);
  // level 3
  phase2Det.push_back(2) ; phase2Det.push_back(2) ; phase2Det.push_back(2); 
  phase2Det.push_back(2) ; phase2Det.push_back(2) ; phase2Det.push_back(5);
  // level 4
  phase2Det.push_back(0) ; phase2Det.push_back(0) ; phase2Det.push_back(0); 
  phase2Det.push_back(0) ; phase2Det.push_back(0) ; phase2Det.push_back(2);
  // level 5
  phase2Det.push_back(-1); phase2Det.push_back(-1); phase2Det.push_back(-1); 
  phase2Det.push_back(-1); phase2Det.push_back(-1); phase2Det.push_back(0);
  
  edm::ParameterSetDescription descDB;
  descDB.add<bool>( "fromDDD", false );
  descDB.addOptional<std::vector<int> >( "detidShifts", presentDet );
  descriptions.add( "trackerNumberingGeometryDB", descDB );

  edm::ParameterSetDescription descPhase1DB;
  descPhase1DB.add<bool>( "fromDDD", false );
  descPhase1DB.addOptional<std::vector<int> >( "detidShifts", phase1Det );
  descriptions.add( "trackerNumbering2017GeometryDB", descPhase1DB );

  edm::ParameterSetDescription descPhase2DB;
  descPhase2DB.add<bool>( "fromDDD", false );
  descPhase2DB.addOptional<std::vector<int> >( "detidShifts", phase2Det );
  descriptions.add( "trackerNumbering2023GeometryDB", descPhase2DB );

  edm::ParameterSetDescription desc;
  desc.add<bool>( "fromDDD", true );
  desc.addOptional<std::vector<int> >( "detidShifts", presentDet );
  descriptions.add( "trackerNumberingGeometry", desc );

  edm::ParameterSetDescription descPhase1;
  descPhase1.add<bool>( "fromDDD", true );
  descPhase1.addOptional<std::vector<int> >( "detidShifts", phase1Det );
  descriptions.add( "trackerNumbering2017Geometry", descPhase1 );

  edm::ParameterSetDescription descPhase2;
  descPhase2.add<bool>( "fromDDD", true );
  descPhase2.addOptional<std::vector<int> >( "detidShifts", phase2Det );
  descriptions.add( "trackerNumbering2023Geometry", descPhase2 );
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
