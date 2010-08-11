#include "Fireworks/Geometry/interface/FWRecoGeometryESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "TGeoManager.h"
#include "TGeoArb8.h"
#include "TGeoMatrix.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"

# define ADD_PIXEL_TOPOLOGY( rawid, detUnit )			\
  const PixelGeomDetUnit* det = dynamic_cast<const PixelGeomDetUnit*>( detUnit ); \
  if( det )							\
  {      							\
    const RectangularPixelTopology* topo = dynamic_cast<const RectangularPixelTopology*>( &det->specificTopology()); \
    m_fwGeometry->idToName[rawid].topology[0] = topo->nrows();	\
    m_fwGeometry->idToName[rawid].topology[1] = topo->ncolumns(); \
  }								\

# define ADD_SISTRIP_TOPOLOGY( rawid, detUnit )			\
  const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>( detUnit ); \
  if( det )                                                     \
  {                                                             \
    const StripTopology* topo = dynamic_cast<const StripTopology*>( &det->specificTopology()); \
    m_fwGeometry->idToName[rawid].topology[0] = topo->nstrips();            \
    m_fwGeometry->idToName[rawid].topology[1] = topo->stripLength();        \
    if( const RadialStripTopology* rtop = dynamic_cast<const RadialStripTopology*>( topo )) \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[2] = rtop->phiPitch();	\
    }                                                                   \
    else if( dynamic_cast<const RectangularStripTopology*>( topo ))     \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[2] = topo->pitch();	\
    }									\
    else if( dynamic_cast<const TrapezoidalStripTopology*>( topo ))     \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[2] = topo->pitch();	\
    }									\
  }                                                                     \
									  
FWRecoGeometryESProducer::FWRecoGeometryESProducer( const edm::ParameterSet& pset )
{
  setWhatProduced( this );
}

FWRecoGeometryESProducer::~FWRecoGeometryESProducer( void )
{}

namespace
{
  /** Create TGeo transformation of GeomDet */
  TGeoCombiTrans* createPlacement( const GeomDet *det )
  {
    // Position of the DetUnit's center
    GlobalPoint pos = det->surface().position();
    TGeoTranslation trans( pos.x(), pos.y(), pos.z());

    // Add the coeff of the rotation matrix
    // with a projection on the basis vectors
    TkRotation<float> detRot = det->surface().rotation();

    TGeoRotation rotation;
    const Double_t matrix[9] = { detRot.xx(), detRot.yx(), detRot.zx(),
				 detRot.xy(), detRot.yy(), detRot.zy(),
				 detRot.xz(), detRot.yz(), detRot.zz() 
    };
    rotation.SetMatrix( matrix );
     
    return new TGeoCombiTrans( trans, rotation );
  }

    std::vector<float> 
  fillPoints( std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end )
  {
    unsigned int index( 0 );
    std::vector<float> points( 24 );
    for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
    {
      assert( index < 24 );
      points[index] = i->x();
      points[++index] = i->y();
      points[++index] = i->z();
      ++index;
    }
    return points;
  }
}

boost::shared_ptr<FWRecoGeometry> 
FWRecoGeometryESProducer::produce( const FWRecoGeometryRecord& record )
{
  using namespace edm;

  m_fwGeometry =  boost::shared_ptr<FWRecoGeometry>( new FWRecoGeometry );

  record.getRecord<GlobalTrackingGeometryRecord>().get( m_geomRecord );
  
  DetId detId( DetId::Tracker, 0 );
  m_trackerGeom = (const TrackerGeometry*) m_geomRecord->slaveGeometry( detId );
  
  record.getRecord<CaloGeometryRecord>().get( m_caloGeom );

  TGeoManager *geom = new TGeoManager( "cmsGeo", "CMS Detector" );
  // NOTE: The default constructor does not create an identity matrix
  if( 0 == gGeoIdentity )
  {
    gGeoIdentity = new TGeoIdentity( "Identity" );
  }

  m_fwGeometry->manager( geom );
  
  // Default material is Vacuum
  TGeoMaterial *matVacuum = new TGeoMaterial( "Vacuum", 0 ,0 ,0 );
  // so is default medium
  TGeoMedium *vacuum = new TGeoMedium( "Vacuum", 1, matVacuum );
  TGeoVolume *top = geom->MakeBox( "CMS", vacuum, 270., 270., 120. );
  
  if( 0 == top )
  {
    return boost::shared_ptr<FWRecoGeometry>();
  }
  geom->SetTopVolume( top );
  // ROOT chokes unless colors are assigned
  top->SetVisibility( kFALSE );
  top->SetLineColor( kBlue );

  addPixelBarrelGeometry( top );
  addPixelForwardGeometry( top );
  addTIBGeometry( top );
  addTIDGeometry( top );
  addTOBGeometry( top );
  addTECGeometry( top );
  addDTGeometry( top );
  addCSCGeometry( top );
  addRPCGeometry( top );

  addCaloGeometry();
  
  geom->CloseGeometry();
  geom->DefaultColors();

  m_nameToShape.clear();
  m_nameToVolume.clear();
  m_nameToMaterial.clear();
  m_nameToMedium.clear();

  return m_fwGeometry;
}

/** Create TGeo shape for GeomDet */
TGeoShape* 
FWRecoGeometryESProducer::createShape( const GeomDet *det )
{
  TGeoShape* shape = 0;

  // Trapezoidal
  const Bounds *b = &((det->surface ()).bounds ());
  if( dynamic_cast<const TrapezoidalPlaneBounds *> (b))
  {
    const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *> (b);
    std::vector< float > par = b2->parameters ();
    
    // These parameters are half-lengths, as in CMSIM/GEANT3
    float hBottomEdge = par [0];
    float hTopEdge    = par [1];
    float thickness   = par [2];
    float apothem     = par [3];

    std::stringstream s;
    s << "T_"
      << hBottomEdge << "_"
      << hTopEdge << "_"
      << thickness << "_"
      << apothem;
    std::string name = s.str();

    // Do not create identical shape,
    // if one already exists
    shape = m_nameToShape[name];
    if( 0 == shape )
    {
      shape = new TGeoTrap(
	name.c_str(),
	thickness,  //dz
	0, 	    //theta
	0, 	    //phi
	apothem,    //dy1
	hBottomEdge,//dx1
	hTopEdge,   //dx2
	0, 	    //alpha1
	apothem,    //dy2
	hBottomEdge,//dx3
	hTopEdge,   //dx4
	0);         //alpha2

      m_nameToShape[name] = shape;
    }
  }
  if( dynamic_cast<const RectangularPlaneBounds *> (b))
  {
    // Rectangular
    float length = det->surface().bounds().length();
    float width = det->surface().bounds ().width();
    float thickness = det->surface().bounds().thickness();

    std::stringstream s;
    s << "R_"
      << width << "_"
      << length << "_"
      << thickness;
    std::string name = s.str();

    // Do not create identical shape,
    // if one already exists
    shape = m_nameToShape[name];
    if( 0 == shape )
    {
      shape = new TGeoBBox( name.c_str(), width / 2., length / 2., thickness / 2. ); // dx, dy, dz

      m_nameToShape[name] = shape;
    }
  }
  
  return shape;
}

/** Create TGeo volume for GeomDet */
TGeoVolume* 
FWRecoGeometryESProducer::createVolume( const std::string& name, const GeomDet *det, const std::string& material )
{
  TGeoVolume* volume = m_nameToVolume[name];
  if( 0 == volume )
  { 
    TGeoShape* solid = createShape( det );
    TGeoMedium* medium = m_nameToMedium[material];
    if( 0 == medium )
    {
      medium = new TGeoMedium( material.c_str(), 0, createMaterial( material ));
      m_nameToMedium[material] = medium;
    }
    if( solid )
    {
      volume = new TGeoVolume( name.c_str(),
			       solid,
			       medium );
      m_nameToVolume[name] = volume;
    }
  }  
  
  return volume;
}

/** Create TGeo material based on its name */
TGeoMaterial*
FWRecoGeometryESProducer::createMaterial( const std::string& name )
{
  TGeoMaterial *material = m_nameToMaterial[name];

  if( material == 0 )
  {
    // FIXME: Do we need to set real parameters of the material?
    material = new TGeoMaterial( name.c_str(),
                                 0, 0, 0 );
    m_nameToMaterial[name] = material;
  }

  return material;
}

const std::string
FWRecoGeometryESProducer::path( TGeoVolume* volume, const std::string& name, int copy )
{
  std::stringstream outs;
  outs << volume->GetName() << "_" << volume->GetNumber() << "/"
       << name << "_" << copy;

  return outs.str();
}

void
FWRecoGeometryESProducer::addCSCGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  if(! m_geomRecord->slaveGeometry( CSCDetId()))
    throw cms::Exception( "FatalError" ) << "Cannnot find CSCGeometry\n";

  const std::vector<GeomDet*>& cscGeom = m_geomRecord->slaveGeometry( CSCDetId())->dets();
  for( std::vector<GeomDet*>::const_iterator it = cscGeom.begin(), itEnd = cscGeom.end(); it != itEnd; ++it )
  {    
    if( CSCChamber* chamber = dynamic_cast<CSCChamber*>(*it))
    {
      unsigned int rawid = chamber->geographicalId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, chamber );
      assembly->AddNode( child, copy, createPlacement( chamber ));
      child->SetLineColor( kBlue );

      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));
    }
    else if( CSCLayer* layer = dynamic_cast<CSCLayer*>(*it))
    {
      unsigned int rawid = layer->geographicalId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, layer );
      assembly->AddNode( child, copy, createPlacement( layer ));
      child->SetLineColor( kBlue );
      
      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

      const CSCStripTopology* stripTopology = layer->geometry()->topology();
      m_fwGeometry->idToName[rawid].topology[0] = stripTopology->yAxisOrientation();
      m_fwGeometry->idToName[rawid].topology[1] = stripTopology->centreToIntersection();
      m_fwGeometry->idToName[rawid].topology[2] = stripTopology->yCentreOfStripPlane();
      m_fwGeometry->idToName[rawid].topology[3] = stripTopology->phiOfOneEdge();
      m_fwGeometry->idToName[rawid].topology[4] = stripTopology->stripOffset();
      m_fwGeometry->idToName[rawid].topology[5] = stripTopology->angularWidth();

      const CSCWireTopology* wireTopology = layer->geometry()->wireTopology();
      m_fwGeometry->idToName[rawid].topology[6] = wireTopology->wireSpacing();
      m_fwGeometry->idToName[rawid].topology[7] = wireTopology->wireAngle();
    }
  }

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addDTGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  //
  // DT chambers geometry
  //
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  const std::vector<GeomDet*>& dtChamberGeom = m_geomRecord->slaveGeometry( DTChamberId())->dets();
  for( std::vector<GeomDet*>::const_iterator it = dtChamberGeom.begin(),
					     end = dtChamberGeom.end(); 
       it != end; ++it )
  {
    if( DTChamber* chamber = dynamic_cast< DTChamber *>(*it))
    {      
      unsigned int rawid = chamber->geographicalId().rawId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, chamber );
      assembly->AddNode( child, copy, createPlacement( chamber ));
      child->SetLineColor( kRed );
      
      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));
    }
  }
  top->AddNode( assembly, copy );

  // Fill in DT super layer parameters
  const std::vector<GeomDet*>& dtSuperLayerGeom = m_geomRecord->slaveGeometry( DTLayerId())->dets();
  for( std::vector<GeomDet*>::const_iterator it = dtSuperLayerGeom.begin(),
					    end = dtSuperLayerGeom.end(); 
       it != end; ++it )
  {
    if( DTSuperLayer* superlayer = dynamic_cast<DTSuperLayer*>(*it))
    {
      unsigned int rawid = superlayer->id().rawId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, superlayer );
      assembly->AddNode( child, copy, createPlacement( superlayer ));
      child->SetLineColor( kBlue );
      
      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

      const BoundPlane& surf = superlayer->surface();
      // Bounds W/H/L:
      m_fwGeometry->idToName[rawid].topology[0] = surf.bounds().width();
      m_fwGeometry->idToName[rawid].topology[1] = surf.bounds().thickness();
      m_fwGeometry->idToName[rawid].topology[2] = surf.bounds().length();
    }
  }

  // Fill in DT layer parameters
  const std::vector<GeomDet*>& dtLayerGeom = m_geomRecord->slaveGeometry( DTSuperLayerId())->dets();
  for( std::vector<GeomDet*>::const_iterator it = dtLayerGeom.begin(),
					    end = dtLayerGeom.end(); 
       it != end; ++it )
  {
    if( DTLayer* layer = dynamic_cast<DTLayer*>(*it))
    {
      unsigned int rawid = layer->id().rawId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, layer );
      assembly->AddNode( child, copy, createPlacement( layer ));
      child->SetLineColor( kBlue );
      
      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

      const DTTopology& topo = layer->specificTopology();
      const BoundPlane& surf = layer->surface();
      // Topology W/H/L:
      m_fwGeometry->idToName[rawid].topology[0] = topo.cellWidth();
      m_fwGeometry->idToName[rawid].topology[1] = topo.cellHeight();
      m_fwGeometry->idToName[rawid].topology[2] = topo.cellLenght();
      m_fwGeometry->idToName[rawid].topology[3] = topo.firstChannel();
      m_fwGeometry->idToName[rawid].topology[4] = topo.lastChannel();
      m_fwGeometry->idToName[rawid].topology[5] = topo.channels();

      // Bounds W/H/L:
      m_fwGeometry->idToName[rawid].topology[6] = surf.bounds().width();
      m_fwGeometry->idToName[rawid].topology[7] = surf.bounds().thickness();
      m_fwGeometry->idToName[rawid].topology[8] = surf.bounds().length();
    }
  }  
}

void
FWRecoGeometryESProducer::addRPCGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  //
  // RPC chambers geometry
  //
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  DetId detId( DetId::Muon, 3 );
  const RPCGeometry* rpcGeom = (const RPCGeometry*) m_geomRecord->slaveGeometry( detId );
  for( std::vector<RPCRoll *>::const_iterator it = rpcGeom->rolls().begin(),
					     end = rpcGeom->rolls().end(); 
       it != end; ++it )
  {
    RPCRoll* roll = (*it);
    if( roll )
    {
      unsigned int rawid = roll->geographicalId().rawId();
      std::stringstream s;
      s << rawid;
      std::string name = s.str();
      
      TGeoVolume* child = createVolume( name, roll );
      assembly->AddNode( child, copy, createPlacement( roll ));
      child->SetLineColor( kYellow );
      
      std::stringstream p;
      p << path( top, iName, copy ) << "/" << name << "_" << copy;
      m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

      const StripTopology& topo = roll->specificTopology();
      m_fwGeometry->idToName[rawid].topology[0] = roll->nstrips();
      m_fwGeometry->idToName[rawid].topology[1] = topo.stripLength();
      m_fwGeometry->idToName[rawid].topology[2] = topo.pitch();
    }
  }
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addPixelBarrelGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXB().begin(),
						    end = m_trackerGeom->detsPXB().end();
       it != end; ++it)
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();

    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

    ADD_PIXEL_TOPOLOGY( rawid, m_trackerGeom->idToDetUnit( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addPixelForwardGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXF().begin(),
						    end = m_trackerGeom->detsPXF().end();
       it != end; ++it )
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();

    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));
    
    ADD_PIXEL_TOPOLOGY( rawid, m_trackerGeom->idToDetUnit( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTIBGeometry( TGeoVolume* top, const std::string& iName, int copy ) 
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTIB().begin(),
						    end = m_trackerGeom->detsTIB().end();
       it != end; ++it )
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();
    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTOBGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTOB().begin(),
						    end = m_trackerGeom->detsTOB().end();
       it != end; ++it )
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();

    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTIDGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTID().begin(),
						    end = m_trackerGeom->detsTID().end();
       it != end; ++it)
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();

    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTECGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTEC().begin(),
						    end = m_trackerGeom->detsTEC().end();
       it != end; ++it )
  {
    DetId detid = ( *it )->geographicalId();
    unsigned int rawid = detid.rawId();

    std::stringstream s;
    s << rawid;
    std::string name = s.str();

    TGeoVolume* child = createVolume( name, *it );
    assembly->AddNode( child, copy, createPlacement( *it ));
    child->SetLineColor( kGreen );

    std::stringstream p;
    p << path( top, iName, copy ) << "/" << name << "_" << copy;
    m_fwGeometry->idToName.insert( std::pair<unsigned int, FWRecoGeom::Info>( rawid, FWRecoGeom::Info( p.str())));

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addCaloGeometry( void )
{
  std::vector<DetId> vid = m_caloGeom->getValidDetIds(); // Calo
  for( std::vector<DetId>::const_iterator it = vid.begin(),
					 end = vid.end();
       it != end; ++it )
  {
    const CaloCellGeometry::CornersVec& cor( m_caloGeom->getGeometry( *it )->getCorners());
    std::vector<float> points = fillPoints( cor.begin(), cor.end());
    m_fwGeometry->idToName[ it->rawId()].points.swap( points );
  }
}
