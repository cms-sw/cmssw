#include "Fireworks/Geometry/interface/FWRecoGeometryESProducer.h"
#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "TGeoManager.h"
#include "TGeoArb8.h"
#include "TGeoMatrix.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"

class FWGeoAssemblyFast : public TGeoVolumeAssembly
{
public:
   FWGeoAssemblyFast( const char *name )
     : TGeoVolumeAssembly( name )
    {}
   
  void AddNodeFast( const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option = "" )
    {
      TGeoVolume::AddNode( vol, copy_no, mat, option );
    }
  
  void Update(void)
    {
      fShape->ComputeBBox();
    }
};

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
    m_fwGeometry->idToName[rawid].topology[0] = 0;            		    \
    m_fwGeometry->idToName[rawid].topology[1] = topo->nstrips();            \
    m_fwGeometry->idToName[rawid].topology[2] = topo->stripLength();        \
    if( const RadialStripTopology* rtop = dynamic_cast<const RadialStripTopology*>( topo )) \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[0] = 1;			\
      m_fwGeometry->idToName[rawid].topology[3] = rtop->yAxisOrientation(); \
      m_fwGeometry->idToName[rawid].topology[4] = rtop->originToIntersection(); \
      m_fwGeometry->idToName[rawid].topology[5] = rtop->phiOfOneEdge(); \
      m_fwGeometry->idToName[rawid].topology[6] = rtop->angularWidth(); \
    }                                                                   \
    else if( dynamic_cast<const RectangularStripTopology*>( topo ))     \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[0] = 2;			\
      m_fwGeometry->idToName[rawid].topology[3] = topo->pitch();	\
    }									\
    else if( dynamic_cast<const TrapezoidalStripTopology*>( topo ))     \
    {                                                                   \
      m_fwGeometry->idToName[rawid].topology[0] = 3;			\
      m_fwGeometry->idToName[rawid].topology[3] = topo->pitch();	\
    }									\
  }                                                                     \
									  
FWRecoGeometryESProducer::FWRecoGeometryESProducer( const edm::ParameterSet& )
  : m_current( -1 ),
    m_material( 0 ),
    m_medium( 0 )
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
  m_material = new TGeoMaterial( "Vacuum", 0 ,0 ,0 );
  // so is default medium
  m_medium = new TGeoMedium( "Vacuum", 1, m_material );
  TGeoVolume *top = geom->MakeBox( "CMS", m_medium, 270., 270., 120. );
  
  if( 0 == top )
  {
    return boost::shared_ptr<FWRecoGeometry>();
  }
  geom->SetTopVolume( top );
  // ROOT chokes unless colors are assigned
  top->SetVisibility( kFALSE );
  top->SetLineColor( kBlue );
  
  // Path to the top volume
  std::stringstream p;
  p << top->GetName() << "_" << top->GetNumber() << "/";
  const std::string path = p.str();
  
  addPixelBarrelGeometry( top, path );
  addPixelForwardGeometry( top, path );
  addTIBGeometry( top, path );
  addTIDGeometry( top, path );
  addTOBGeometry( top, path );
  addTECGeometry( top, path );
  addDTGeometry( top, path );
  addCSCGeometry( top, path );
  addRPCGeometry( top, path );
  top->GetShape()->ComputeBBox();
  
  addCaloGeometry();
  
  geom->CloseGeometry();

  m_fwGeometry->idToName.resize( m_current + 1 );
  std::vector<FWRecoGeom::Info>( m_fwGeometry->idToName ).swap( m_fwGeometry->idToName );
  std::sort( m_fwGeometry->idToName.begin(), m_fwGeometry->idToName.end());

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

    shape = new TGeoTrap(
      "Trap",
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
  }
  if( dynamic_cast<const RectangularPlaneBounds *> (b))
  {
    // Rectangular
    float length = det->surface().bounds().length();
    float width = det->surface().bounds ().width();
    float thickness = det->surface().bounds().thickness();

    shape = new TGeoBBox( "Box", width / 2., length / 2., thickness / 2. ); // dx, dy, dz
  }
  
  return shape;
}

/** Create TGeo volume for GeomDet */
TGeoVolume* 
FWRecoGeometryESProducer::createVolume( unsigned int rawid, const GeomDet *det )
{
  TGeoVolume* volume = 0;
  TGeoShape* solid = createShape( det );
  if( solid )
  {
    volume = new TGeoVolume( Form( "%u", rawid),
			     solid,
			     m_medium );
  }
  
  return volume;
}

void
FWRecoGeometryESProducer::addCSCGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  DetId detId( DetId::Muon, 2 ); 
  const CSCGeometry* cscGeometry = (const CSCGeometry*) m_geomRecord->slaveGeometry( detId );
  for( std::vector<CSCChamber*>::const_iterator it = cscGeometry->chambers().begin(),
					       end = cscGeometry->chambers().end(); 
       it != end; ++it )
  {
    const CSCChamber *chamber = *it;
    
    if( chamber )
    {
      unsigned int rawid = chamber->geographicalId();
      
      TGeoVolume* child = createVolume( rawid, chamber );
      assembly->AddNodeFast( child, copy, createPlacement( chamber ));
      child->SetLineColor( kBlue );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      /* unsigned int current = */ insert_id( rawid, s.str());
      //
      // CSC layers geometry
      //
      for( std::vector< const CSCLayer* >::const_iterator lit = chamber->layers().begin(),
							 lend = chamber->layers().end(); 
	   lit != lend; ++lit )
      {
	const CSCLayer* layer = *lit;
    
	if( layer )
	{
	  unsigned int rawid = layer->geographicalId();

	  TGeoVolume* child = createVolume( rawid, layer );
	  assembly->AddNodeFast( child, copy, createPlacement( layer ));
	  child->SetLineColor( kBlue );
      
	  s.clear();
	  s.str( "" );
	  s << path << iName << "_1/" << rawid << "_1";
	  unsigned int current = insert_id( rawid, s.str());

	  const CSCStripTopology* stripTopology = layer->geometry()->topology();
	  m_fwGeometry->idToName[current].topology[0] = stripTopology->yAxisOrientation();
	  m_fwGeometry->idToName[current].topology[1] = stripTopology->centreToIntersection();
	  m_fwGeometry->idToName[current].topology[2] = stripTopology->yCentreOfStripPlane();
	  m_fwGeometry->idToName[current].topology[3] = stripTopology->phiOfOneEdge();
	  m_fwGeometry->idToName[current].topology[4] = stripTopology->stripOffset();
	  m_fwGeometry->idToName[current].topology[5] = stripTopology->angularWidth();

	  const CSCWireTopology* wireTopology = layer->geometry()->wireTopology();
	  m_fwGeometry->idToName[current].topology[6] = wireTopology->wireSpacing();
	  m_fwGeometry->idToName[current].topology[7] = wireTopology->wireAngle();
	}
      }
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addDTGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  DetId detId( DetId::Muon, 1 );
  const DTGeometry* dtGeometry = (const DTGeometry*) m_geomRecord->slaveGeometry( detId );

  //
  // DT chambers geometry
  //
  for( std::vector<DTChamber *>::const_iterator it = dtGeometry->chambers().begin(),
					       end = dtGeometry->chambers().end(); 
       it != end; ++it )
  {
    const DTChamber *chamber = *it;
    
    if( chamber )
    {
      unsigned int rawid = chamber->geographicalId().rawId();
      
      TGeoVolume* child = createVolume( rawid, chamber );
      assembly->AddNodeFast( child, copy, createPlacement( chamber ));
      child->SetLineColor( kRed );
      
      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      /* unsigned int current = */ insert_id( rawid, s.str());
    }
  }

  // Fill in DT layer parameters
  for( std::vector<DTLayer*>::const_iterator it = dtGeometry->layers().begin(),
					    end = dtGeometry->layers().end(); 
       it != end; ++it )
  {
    const DTLayer* layer = *it;
     
    if( layer )
    {
      unsigned int rawid = layer->id().rawId();
      
      TGeoVolume* child = createVolume( rawid, layer );
      assembly->AddNodeFast( child, copy, createPlacement( layer ));
      child->SetLineColor( kBlue );
      
      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      const DTTopology& topo = layer->specificTopology();
      const BoundPlane& surf = layer->surface();
      // Topology W/H/L:
      m_fwGeometry->idToName[current].topology[0] = topo.cellWidth();
      m_fwGeometry->idToName[current].topology[1] = topo.cellHeight();
      m_fwGeometry->idToName[current].topology[2] = topo.cellLenght();
      m_fwGeometry->idToName[current].topology[3] = topo.firstChannel();
      m_fwGeometry->idToName[current].topology[4] = topo.lastChannel();
      m_fwGeometry->idToName[current].topology[5] = topo.channels();

      // Bounds W/H/L:
      m_fwGeometry->idToName[current].topology[6] = surf.bounds().width();
      m_fwGeometry->idToName[current].topology[7] = surf.bounds().thickness();
      m_fwGeometry->idToName[current].topology[8] = surf.bounds().length();
    }
  }  
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addRPCGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  //
  // RPC rolls geometry
  //
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
      
      TGeoVolume* child = createVolume( rawid, roll );
      assembly->AddNodeFast( child, copy, createPlacement( roll ));
      child->SetLineColor( kYellow );
      
      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      const StripTopology& topo = roll->specificTopology();
      m_fwGeometry->idToName[current].topology[0] = roll->nstrips();
      m_fwGeometry->idToName[current].topology[1] = topo.stripLength();
      m_fwGeometry->idToName[current].topology[2] = topo.pitch();
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addPixelBarrelGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXB().begin(),
						    end = m_trackerGeom->detsPXB().end();
       it != end; ++it)
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();

      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      ADD_PIXEL_TOPOLOGY( current, m_trackerGeom->idToDetUnit( detid ));
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addPixelForwardGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXF().begin(),
						    end = m_trackerGeom->detsPXF().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();

      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());
    
      ADD_PIXEL_TOPOLOGY( current, m_trackerGeom->idToDetUnit( detid ));
    }
  }
  
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTIBGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy ) 
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTIB().begin(),
						    end = m_trackerGeom->detsTIB().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
    
      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTOBGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTOB().begin(),
						    end = m_trackerGeom->detsTOB().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();

      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTIDGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTID().begin(),
						    end = m_trackerGeom->detsTID().end();
       it != end; ++it)
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();

      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
  assembly->Update();

  top->AddNode( assembly, copy );
}

void
FWRecoGeometryESProducer::addTECGeometry( TGeoVolume* top, const std::string& path, const std::string& iName, int copy )
{
  FWGeoAssemblyFast *assembly = new FWGeoAssemblyFast( iName.c_str());
  std::stringstream s;
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTEC().begin(),
						    end = m_trackerGeom->detsTEC().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();

      TGeoVolume* child = createVolume( rawid, det );
      assembly->AddNodeFast( child, copy, createPlacement( det ));
      child->SetLineColor( kGreen );

      s.clear();
      s.str( "" );
      s << path << iName << "_1/" << rawid << "_1";
      unsigned int current = insert_id( rawid, s.str());

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
  assembly->Update();

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
    unsigned int id = insert_id( it->rawId());
    fillPoints( id, cor.begin(), cor.end());
  }
}

unsigned int
FWRecoGeometryESProducer::insert_id( unsigned int rawid, const std::string& name )
{
  ++m_current;
  m_fwGeometry->idToName[m_current].id = rawid;
  m_fwGeometry->idToName[m_current].name = name;
  assert( m_current >= 0 );
  assert( m_current < 260000 );
  
  return m_current;
}

void
FWRecoGeometryESProducer::fillPoints( unsigned int id, std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end )
{
  unsigned int index( 0 );
  for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
  {
    assert( index < 24 );
    m_fwGeometry->idToName[id].points[index] = i->x();
    m_fwGeometry->idToName[id].points[++index] = i->y();
    m_fwGeometry->idToName[id].points[++index] = i->z();
    ++index;
  }
}
