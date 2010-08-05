#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
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

#include "CLHEP/Units/GlobalSystemOfUnits.h"

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
    m_idToName[rawid].topology[0] = topo->nrows();		\
    m_idToName[rawid].topology[1] = topo->ncolumns();		\
    m_idToName[rawid].topology[2] = topo->pitch().first;        \
    m_idToName[rawid].topology[3] = topo->pitch().second;       \
  }								\

# define ADD_SISTRIP_TOPOLOGY( rawid, detUnit )			\
  const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>( detUnit ); \
  if( det )                                                     \
  {                                                             \
    const StripTopology* topo = dynamic_cast<const StripTopology*>( &det->specificTopology()); \
    m_idToName[rawid].topology[0] = topo->nstrips();            \
  }                                                             \


class DumpRecoGeom : public edm::EDAnalyzer
{
public:
  explicit DumpRecoGeom( const edm::ParameterSet& pSet );
  ~DumpRecoGeom( void ) {}

private:
  virtual void analyze( const edm::Event& event, const edm::EventSetup& eventSetup );
  virtual void beginJob( void );
  virtual void endJob( void );

  TGeoCombiTrans* createPlacement( const GeomDet *det );
  TGeoShape* createShape( const GeomDet *det );
  TGeoVolume* createVolume( const std::string& name, const GeomDet *det, const std::string& matname = "Air" );
  TGeoMaterial* createMaterial( const std::string& name );
  const std::string path( TGeoVolume* top, const std::string& name, int copy );

  void addCSCGeometry( TGeoVolume* top, const std::string& name = "CSC", int copy = 1 );
  void addDTGeometry( TGeoVolume* top, const std::string& name = "DT", int copy = 1 );
  void addRPCGeometry( TGeoVolume* top, const std::string& name = "RPC", int copy = 1 );
  void addPixelBarrelGeometry( TGeoVolume* top, const std::string& name = "PixelBarrel", int copy = 1 );
  void addPixelForwardGeometry( TGeoVolume* top, const std::string& name = "PixelForward", int copy = 1 );
  void addTIBGeometry( TGeoVolume* top, const std::string& name = "TIB", int copy = 1 );
  void addTOBGeometry( TGeoVolume* top, const std::string& name = "TOB", int copy = 1 );
  void addTIDGeometry( TGeoVolume* top, const std::string& name = "TID", int copy = 1 );
  void addTECGeometry( TGeoVolume* top, const std::string& name = "TEC", int copy = 1 );
  void addCaloGeometry( void );

  int m_level;
  
  struct Info
  {
    std::string name;
    Float_t points[24]; // x1,y1,z1...x8,y8,z8
    Float_t topology[9]; 
    Info( const std::string& iname )
      : name( iname )
      {
	init();
      }
    Info( void )
      {
	init();
      }
    void
    init( void )
      {
	for( unsigned int i = 0; i < 24; ++i ) points[i] = 0;
	for( unsigned int i = 0; i < 9; ++i ) topology[i] = 0;
      }
    void
    fillPoints( std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end )
      {
	 unsigned int index( 0 );
	 for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
	 {
	    assert( index < 8 );
	    points[index*3] = i->x();
	    points[index*3+1] = i->y();
	    points[index*3+2] = i->z();
	    ++index;
	 }
      }
  };

  std::map<std::string, TGeoShape*>    m_nameToShape;
  std::map<std::string, TGeoVolume*>   m_nameToVolume;
  std::map<std::string, TGeoMaterial*> m_nameToMaterial;
  std::map<std::string, TGeoMedium*>   m_nameToMedium;
  std::map<unsigned int, Info>         m_idToName;

  edm::ESHandle<TrackerGeometry> 	m_trackerGeom;
  edm::ESHandle<CaloGeometry>           m_caloGeom;
  edm::ESHandle<CSCGeometry>            m_cscGeom;
  edm::ESHandle<DTGeometry>             m_dtGeom;
  edm::ESHandle<RPCGeometry>            m_rpcGeom;
};

DumpRecoGeom::DumpRecoGeom( const edm::ParameterSet& iConfig )
  : m_level( iConfig.getUntrackedParameter<int>( "level", 1 ))
{}

void
DumpRecoGeom::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  using namespace edm;

  eventSetup.get<TrackerDigiGeometryRecord>().get( m_trackerGeom );
  eventSetup.get<CaloGeometryRecord>().get( m_caloGeom );
  eventSetup.get<MuonGeometryRecord>().get( m_cscGeom );
  eventSetup.get<MuonGeometryRecord>().get( m_dtGeom );
  eventSetup.get<MuonGeometryRecord>().get( m_rpcGeom );

  std::auto_ptr<TGeoManager> geom( new TGeoManager( "cmsGeo", "CMS Detector" ));
  // NOTE: The default constructor does not create an identity matrix
  if( 0 == gGeoIdentity )
  {
    gGeoIdentity = new TGeoIdentity( "Identity" );
  }

  // Default material is Vacuum
  TGeoMaterial *matVacuum = new TGeoMaterial( "Vacuum", 0 ,0 ,0 );
  // so is default medium
  TGeoMedium *vacuum = new TGeoMedium( "Vacuum", 1, matVacuum );
  TGeoVolume *top = geom->MakeBox( "CMS", vacuum, 270., 270., 120. );
  
  if( 0 == top )
  {
    return;
  }
  geom->SetTopVolume( top );
  // ROOT chokes unless colors are assigned
  top->SetVisibility( kFALSE );
  top->SetLineColor( kBlue );

  addCSCGeometry( top );
  addDTGeometry( top );
  addRPCGeometry( top );
  addPixelBarrelGeometry( top );
  addPixelForwardGeometry( top );
  addTIBGeometry( top );
  addTOBGeometry( top );
  addTIDGeometry( top );
  addTECGeometry( top );
  addCaloGeometry();
  
  geom->CloseGeometry();

  std::stringstream s;
  s << "cmsRecoGeom" << m_level << ".root";
  TFile file( s.str().c_str(), "RECREATE" );
   
  TTree *tree = new TTree( "idToGeo", "Raw detector id association with geometry" );
  UInt_t v_id;
  TString *v_path( new TString );
  char v_name[1000];
  Float_t v_vertex[24];
  Float_t v_params[9];
//   TGeoHMatrix* v_matrix( new TGeoHMatrix );
  
// An attempt to cache shapes and matrices.  
//   TObject* v_volume( new TObject );
//   TObject* v_shape( new TObject );

  tree->SetBranchStyle( 0 );
  tree->Branch( "id", &v_id, "id/i" );
  tree->Branch( "path", &v_name, "path/C" );
//   tree->Branch( "volume", "TObject", &v_volume );
//   tree->Branch( "shape", "TObject", &v_shape );
  tree->Branch( "points", &v_vertex, "points[24]/F" );
  tree->Branch( "topology", &v_params, "topology[9]/F" );
//   tree->Branch( "matrix", "TGeoHMatrix", &v_matrix );
  for( std::map<unsigned int, Info>::const_iterator it = m_idToName.begin(),
						   end = m_idToName.end();
       it != end; ++it )
  {
    v_id = it->first;
    *v_path = it->second.name.c_str();
    for( unsigned int i = 0; i < 24; ++i )
      v_vertex[i] = it->second.points[i];
    for( unsigned int i = 0; i < 9; ++i )
      v_params[i] = it->second.topology[i];
    strcpy( v_name, it->second.name.c_str());
//     geom->cd( *v_path );
//     v_matrix = geom->GetCurrentMatrix();
//     v_volume = geom->GetCurrentVolume();
//     v_shape = geom->GetCurrentVolume()->GetShape();
    tree->Fill();
  }
  file.WriteTObject( &*geom );
  file.WriteTObject( tree );
  file.Close();
}

void 
DumpRecoGeom::beginJob( void )
{}

void 
DumpRecoGeom::endJob( void )
{}

/** Create TGeo transformation of GeomDet */
TGeoCombiTrans*
DumpRecoGeom::createPlacement( const GeomDet *det )
{
  // Position of the DetUnit's center
  float posx = det->surface().position().x()/mm;
  float posy = det->surface().position().y()/mm;
  float posz = det->surface().position().z()/mm;

  TGeoTranslation trans( posx, posy, posz );

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

/** Create TGeo shape for GeomDet */
TGeoShape*
DumpRecoGeom::createShape( const GeomDet *det )
{    
  TGeoShape* shape = 0;

  // Trapezoidal
  const Bounds *b = &((det->surface ()).bounds ());
  if( dynamic_cast<const TrapezoidalPlaneBounds *> (b))
  {
    const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *> (b);
    std::vector< float > par = b2->parameters ();
    
    // These parameters are half-lengths, as in CMSIM/GEANT3
    float hBottomEdge = par [0]/mm;
    float hTopEdge    = par [1]/mm;
    float thickness   = par [2]/mm;
    float apothem     = par [3]/mm;

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
    float length = det->surface().bounds().length()/mm;
    float width = det->surface().bounds ().width()/mm;
    float thickness = det->surface().bounds().thickness()/mm;

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
DumpRecoGeom::createVolume( const std::string& name, const GeomDet *det, const std::string& material )
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
DumpRecoGeom::createMaterial( const std::string& name )
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
DumpRecoGeom::path( TGeoVolume* volume, const std::string& name, int copy )
{
  std::stringstream outs;
  outs << volume->GetName() << "_" << volume->GetNumber() << "/"
       << name << "_" << copy;

  return outs.str();
}

void
DumpRecoGeom::addCSCGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  //
  // CSC chambers geometry
  //
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( std::vector<CSCChamber*>::const_iterator it = m_cscGeom->chambers().begin(),
					       end = m_cscGeom->chambers().end(); 
       it != end; ++it )
  {
    const CSCChamber *chamber = *it;
    
    if( chamber )
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
      m_idToName[rawid] = Info( p.str());
    }
  }
  //
  // CSC layers geometry
  //
  for( std::vector<CSCLayer*>::const_iterator it = m_cscGeom->layers().begin(),
					     end = m_cscGeom->layers().end(); 
       it != end; ++it )
  {
    const CSCLayer* layer = *it;
    
    if( layer )
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
      m_idToName[rawid] = Info( p.str());

      const CSCStripTopology* stripTopology = layer->geometry()->topology();
      m_idToName[rawid].topology[0] = stripTopology->yAxisOrientation();
      m_idToName[rawid].topology[1] = stripTopology->centreToIntersection();
      m_idToName[rawid].topology[2] = stripTopology->yCentreOfStripPlane();
      m_idToName[rawid].topology[3] = stripTopology->phiOfOneEdge();
      m_idToName[rawid].topology[4] = stripTopology->stripOffset();
      m_idToName[rawid].topology[5] = stripTopology->angularWidth();

      const CSCWireTopology* wireTopology = layer->geometry()->wireTopology();
      m_idToName[rawid].topology[6] = wireTopology->wireSpacing();
      m_idToName[rawid].topology[7] = wireTopology->wireAngle();
    }
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addDTGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  //
  // DT chambers geometry
  //
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( std::vector<DTChamber *>::const_iterator it = m_dtGeom->chambers().begin(),
					       end = m_dtGeom->chambers().end(); 
       it != end; ++it )
  {
    const DTChamber *chamber = *it;
    
    if( chamber )
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
      m_idToName[rawid] = Info( p.str());
    }
  }
  top->AddNode( assembly, copy );

  // Fill in DT super layer parameters
  for( std::vector<DTSuperLayer*>::const_iterator it = m_dtGeom->superLayers().begin(),
						 end = m_dtGeom->superLayers().end(); 
       it != end; ++it )
  {
    const DTSuperLayer* superlayer = *it;
     
    if( superlayer )
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
      m_idToName[rawid] = Info( p.str());

      const BoundPlane& surf = superlayer->surface();
      // Bounds W/H/L:
      m_idToName[rawid].topology[0] = surf.bounds().width();
      m_idToName[rawid].topology[1] = surf.bounds().thickness();
      m_idToName[rawid].topology[2] = surf.bounds().length();
    }
  }

  // Fill in DT layer parameters
  for( std::vector<DTLayer*>::const_iterator it = m_dtGeom->layers().begin(),
					    end = m_dtGeom->layers().end(); 
       it != end; ++it )
  {
    const DTLayer* layer = *it;
     
    if( layer )
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
      m_idToName[rawid] = Info( p.str());

      const DTTopology& topo = layer->specificTopology();
      const BoundPlane& surf = layer->surface();
      // Topology W/H/L:
      m_idToName[rawid].topology[0] = topo.cellWidth();
      m_idToName[rawid].topology[1] = topo.cellHeight();
      m_idToName[rawid].topology[2] = topo.cellLenght();
      m_idToName[rawid].topology[3] = topo.firstChannel();
      m_idToName[rawid].topology[4] = topo.lastChannel();
      m_idToName[rawid].topology[5] = topo.channels();

      // Bounds W/H/L:
      m_idToName[rawid].topology[6] = surf.bounds().width();
      m_idToName[rawid].topology[7] = surf.bounds().thickness();
      m_idToName[rawid].topology[8] = surf.bounds().length();
    }
  }  
}

void
DumpRecoGeom::addRPCGeometry( TGeoVolume* top, const std::string& iName, int copy )
{
  //
  // RPC chambers geometry
  //
  TGeoVolume *assembly = new TGeoVolumeAssembly( iName.c_str());
  for( std::vector<RPCRoll *>::const_iterator it = m_rpcGeom->rolls().begin(),
					     end = m_rpcGeom->rolls().end(); 
       it != end; ++it )
  {
    const RPCRoll *roll = *it;
    
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
      m_idToName[rawid] = Info( p.str());

      const StripTopology& topo = roll->specificTopology();
      m_idToName[rawid].topology[0] = roll->nstrips();
      m_idToName[rawid].topology[1] = topo.stripLength();
      m_idToName[rawid].topology[2] = topo.pitch();
    }
  }
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addPixelBarrelGeometry( TGeoVolume* top, const std::string& iName, int copy )
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
    m_idToName[rawid] = Info( p.str());

    ADD_PIXEL_TOPOLOGY( rawid, m_trackerGeom->idToDetUnit( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addPixelForwardGeometry( TGeoVolume* top, const std::string& iName, int copy )
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
    m_idToName[rawid] = Info( p.str());
    
    ADD_PIXEL_TOPOLOGY( rawid, m_trackerGeom->idToDetUnit( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addTIBGeometry( TGeoVolume* top, const std::string& iName, int copy ) 
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
    m_idToName[rawid] = Info( p.str());

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addTOBGeometry( TGeoVolume* top, const std::string& iName, int copy )
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
    m_idToName[rawid] = Info( p.str());

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addTIDGeometry( TGeoVolume* top, const std::string& iName, int copy )
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
    m_idToName[rawid] = Info( p.str());

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addTECGeometry( TGeoVolume* top, const std::string& iName, int copy )
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
    m_idToName[rawid] = Info( p.str());

    ADD_SISTRIP_TOPOLOGY( rawid, m_trackerGeom->idToDet( detid ));
  }
  
  top->AddNode( assembly, copy );
}

void
DumpRecoGeom::addCaloGeometry( void )
{
  std::vector<DetId> vid = m_caloGeom->getValidDetIds(); // Calo
  for( std::vector<DetId>::const_iterator it = vid.begin(),
					 end = vid.end();
       it != end; ++it )
  {
    const CaloCellGeometry::CornersVec& cor( m_caloGeom->getGeometry( *it )->getCorners());
    m_idToName[it->rawId()].fillPoints( cor.begin(), cor.end());
  }
}

DEFINE_FWK_MODULE( DumpRecoGeom );
