#include "Fireworks/Geometry/interface/FWTGeoRecoGeometryESProducer.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
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
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "TGeoManager.h"
#include "TGeoArb8.h"
#include "TGeoMatrix.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"

FWTGeoRecoGeometryESProducer::FWTGeoRecoGeometryESProducer( const edm::ParameterSet& /*pset*/ ):
m_dummyMedium(0)
{
  setWhatProduced( this );
}

FWTGeoRecoGeometryESProducer::~FWTGeoRecoGeometryESProducer( void )
{}

namespace
{

void AddLeafNode(TGeoVolume* mother, TGeoVolume* daughter, const char* name, TGeoMatrix* mtx )
{
   int n = mother->GetNdaughters();
   mother->AddNode(daughter, 1, mtx);
   mother->GetNode(n)->SetName(name);
}

  /** Create TGeo transformation of GeomDet */
TGeoCombiTrans* createPlacement( const GeomDet *det )
{
   // Position of the DetUnit's center
   float posx = det->surface().position().x();
   float posy = det->surface().position().y();
   float posz = det->surface().position().z();

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



TGeoMatrix* createCaloPlacement( const CaloCellGeometry* cell)
{
   CaloCellGeometry::Tr3D  tr;
   cell->getTransform(tr, 0);
        
   TGeoTranslation trans( tr.getTranslation().x(), tr.getTranslation().y(), tr.getTranslation().z());

   TGeoRotation rotation;
   CLHEP::HepRotation  detRot = tr.getRotation();
   const Double_t matrix[9] = { detRot.xx(), detRot.yx(), detRot.zx(),
                                detRot.xy(), detRot.yy(), detRot.zy(),
                                detRot.xz(), detRot.yz(), detRot.zz() 
   };


   rotation.SetMatrix( matrix );
   TGeoMatrix* res = new TGeoCombiTrans( trans, rotation );
   res->Print();
   return res;
}
}

TGeoVolume* FWTGeoRecoGeometryESProducer::GetDaughter(TGeoVolume* mother, const char* prefix, int id)
{
   TGeoVolume* res = 0;
   if (mother->GetNdaughters()) { 
      TGeoNode* n = mother->FindNode(Form("%s_%d_1", prefix, id));
      if ( n ) res = n->GetVolume();
   }

   if (!res) {
      res = new TGeoVolumeAssembly( Form("%s_%d", prefix, id ));
      res->SetMedium(m_dummyMedium);
      mother->AddNode(res, 1);
   }

   return res;
}

boost::shared_ptr<FWTGeoRecoGeometry> 
FWTGeoRecoGeometryESProducer::produce( const FWTGeoRecoGeometryRecord& record )
{
   using namespace edm;

  m_fwGeometry = boost::shared_ptr<FWTGeoRecoGeometry>( new FWTGeoRecoGeometry );

   record.getRecord<GlobalTrackingGeometryRecord>().get( m_geomRecord );
  
   DetId detId( DetId::Tracker, 0 );
   m_trackerGeom = (const TrackerGeometry*) m_geomRecord->slaveGeometry( detId );
  
   record.getRecord<CaloGeometryRecord>().get( m_caloGeom );

   TGeoManager* geom = new TGeoManager( "cmsGeo", "CMS Detector" );
   if( 0 == gGeoIdentity )
   {
      gGeoIdentity = new TGeoIdentity( "Identity" );
   }

  m_fwGeometry->manager( geom );
  
   // Default material is Vacuum
   TGeoMaterial *vacuum = new TGeoMaterial( "Vacuum", 0 ,0 ,0 );
   m_dummyMedium = new TGeoMedium( "reco", 0, vacuum);
   // so is default medium
   TGeoVolume *top = geom->MakeBox( "CMS", m_dummyMedium, 270., 270., 120. );
  
   if( 0 == top )
   {
     return boost::shared_ptr<FWTGeoRecoGeometry>();
   }
   geom->SetTopVolume( top );
   // ROOT chokes unless colors are assigned
   top->SetVisibility( kFALSE );
   top->SetLineColor( kBlue );
   
   addPixelBarrelGeometry();
   addPixelForwardGeometry();
   addTIBGeometry();
   addTIDGeometry();
   addTOBGeometry();
   addTECGeometry();
   addDTGeometry();
   addCSCGeometry();
   addRPCGeometry();
   try {
      addGEMGeometry();
   }
   catch ( cms::Exception& exception ) {
   edm::LogWarning("FWRecoGeometryProducerException")
     << "addGEMGeometry() Exception caught while building GEM geometry: " << exception.what()
     << std::endl; 
   }

   //  addEcalCaloGeometry();
   geom->CloseGeometry();

   m_nameToShape.clear();
   m_shapeToVolume.clear();
   m_nameToMaterial.clear();
   m_nameToMedium.clear();

  return m_fwGeometry;
}

/** Create TGeo shape for GeomDet */
TGeoShape* 
FWTGeoRecoGeometryESProducer::createShape( const GeomDet *det )
{
  TGeoShape* shape = 0;

  // Trapezoidal
  const Bounds *b = &((det->surface ()).bounds ());
  const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *> (b);
  if( b2 )
  {
      std::array< const float, 4 > const & par = b2->parameters ();
    
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
  if( dynamic_cast<const RectangularPlaneBounds *> (b) != 0 )
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
FWTGeoRecoGeometryESProducer::createVolume( const std::string& name, const GeomDet *det, const std::string& material )
{
   TGeoShape* solid = createShape( det );

   std::map<TGeoShape*, TGeoVolume*>::iterator vIt = m_shapeToVolume.find(solid);
   if (vIt != m_shapeToVolume.end()) return  vIt->second;
   

   TGeoMedium* medium = m_nameToMedium[material];
   if( 0 == medium )
   {
      medium = new TGeoMedium( material.c_str(), 0, createMaterial( material ));
      m_nameToMedium[material] = medium;
   }
   TGeoVolume* volume = new TGeoVolume( name.c_str(),solid, medium);

   m_shapeToVolume[solid] = volume;

   return volume;
}

/** Create TGeo material based on its name */
TGeoMaterial*
FWTGeoRecoGeometryESProducer::createMaterial( const std::string& name )
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


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


void
FWTGeoRecoGeometryESProducer::addPixelBarrelGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("PXB");

   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXB().begin(),
           end = m_trackerGeom->detsPXB().end();
        it != end; ++it)
   {
       DetId detid = ( *it )->geographicalId();
       unsigned int rawid = detid.rawId();

       PXBDetId xx(rawid);
       std::string name = Form("PXB Ly:%d, Md:%d Ld:%d ", xx.layer(), xx.module(), xx.layer());
       TGeoVolume* child = createVolume( name, *it );
       child->SetLineColor( kGreen );

       TGeoVolume* holder  = GetDaughter(assembly, "Layer", xx.layer());
       holder = GetDaughter(holder, "Module", xx.module());
                                 
       AddLeafNode(holder, child, name.c_str(), createPlacement( *it ));
   }
  

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}
//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addPixelForwardGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("PXF");
   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXF().begin(),
           end = m_trackerGeom->detsPXF().end();
        it != end; ++it )
   {
      PXFDetId detid = ( *it )->geographicalId();

      std::stringstream s;
      s << detid;
      std::string name = s.str();

      TGeoVolume* child = createVolume( name, *it );


      TGeoVolume* holder  = GetDaughter(assembly, "Side", detid.side());
      holder = GetDaughter(holder, "Disk", detid.disk());
      holder = GetDaughter(holder, "Blade", detid.blade());
      holder = GetDaughter(holder, "Panel", detid.panel());
   
      // holder->AddNode( child, 1, createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( *it ));
      child->SetLineColor( kGreen );

   }
  

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}

//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addTIBGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("TIB");
   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTIB().begin(),
           end = m_trackerGeom->detsTIB().end();
        it != end; ++it )
   {
      TIBDetId detid(( *it )->geographicalId());

      std::stringstream s;
      s << detid;
      std::string name = s.str();

      TGeoVolume* child = createVolume( name, *it );

      TGeoVolume* holder  = GetDaughter(assembly, "Module", detid.module());
      holder = GetDaughter(holder, "Order", detid.order());
      holder = GetDaughter(holder, "Side", detid.side());
      // holder->AddNode( child, 1, createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( *it ));
      child->SetLineColor( kGreen );
   }
  
   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}

//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addTIDGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("TID");

   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTID().begin(),
           end = m_trackerGeom->detsTID().end();
        it != end; ++it)
   {
      TIDDetId detid = ( *it )->geographicalId();

      std::stringstream s;
      s << detid;
      std::string name = s.str();

      TGeoVolume* child = createVolume( name, *it );
      TGeoVolume* holder  = GetDaughter(assembly, "Side", detid.side());
      holder = GetDaughter(holder, "Wheel", detid.wheel());
      holder = GetDaughter(holder, "Ring", detid.ring());
      //  holder->AddNode( child, 1, createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( *it ));
   
      child->SetLineColor( kGreen );
   }

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}

//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addTOBGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("TOB");

   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTOB().begin(),
           end = m_trackerGeom->detsTOB().end();
        it != end; ++it )
   {
      TOBDetId detid(( *it )->geographicalId());

      std::stringstream s;
      s << detid;
      std::string name = s.str();

      TGeoVolume* child = createVolume( name, *it );
      TGeoVolume* holder  = GetDaughter(assembly, "Rod", detid.rodNumber());
      holder = GetDaughter(holder, "Side", detid.side());
      holder = GetDaughter(holder, "Module", detid.moduleNumber());
      //holder->AddNode( child, 1, createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( *it ));
   
      child->SetLineColor( kGreen );
   }

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}
//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addTECGeometry()
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("TEC");

   for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTEC().begin(),
           end = m_trackerGeom->detsTEC().end();
        it != end; ++it )
   {
      TECDetId detid = ( *it )->geographicalId();

      std::stringstream s;
      s << detid;
      std::string name = s.str();

      TGeoVolume* child = createVolume( name, *it );

      TGeoVolume* holder  = GetDaughter(assembly, "Order", detid.order());
      holder = GetDaughter(holder, "Ring", detid.ring());
      holder = GetDaughter(holder, "Module", detid.module());
      // holder->AddNode( child, 1, createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( *it ));
      child->SetLineColor( kGreen );
   }

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}
//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addDTGeometry(  )
{
   //
   // DT chambers geometry
   //
   TGeoVolume *assembly = new TGeoVolumeAssembly( "DT");

   auto const & dtChamberGeom = m_geomRecord->slaveGeometry( DTChamberId())->dets();
   for( auto it = dtChamberGeom.begin(),
           end = dtChamberGeom.end(); 
        it != end; ++it )
   {
      if( auto chamber = dynamic_cast< const DTChamber *>(*it))
      {      
         DTChamberId detid = chamber->geographicalId();
         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, chamber );
         TGeoVolume* holder  = GetDaughter(assembly, "Wheel", detid.wheel());
         holder = GetDaughter(holder, "Station", detid.station());
         holder = GetDaughter(holder, "Sector", detid.sector());
   
         //   holder->AddNode( child, 1, createPlacement( chamber ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( chamber));
         child->SetLineColor( kRed );
      }
   }

   // Fill in DT super layer parameters
   auto const & dtSuperLayerGeom = m_geomRecord->slaveGeometry( DTSuperLayerId())->dets();
   for( auto it = dtSuperLayerGeom.begin(),
           end = dtSuperLayerGeom.end(); 
        it != end; ++it )
   {
      if( auto * superlayer = dynamic_cast<const DTSuperLayer*>(*it))
      {
         //         DetId detidx = superlayer->geographicalId();
         DTSuperLayerId detid( DetId(superlayer->geographicalId()));
         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, superlayer );

         TGeoVolume* holder  = GetDaughter(assembly, "Wheel", detid.wheel());
         holder = GetDaughter(holder, "Station", detid.station());
         holder = GetDaughter(holder, "Sector", detid.sector());
         holder = GetDaughter(holder, "SuperLayer", detid.superlayer());
         // holder = GetDaughter(holder, "Layer", detid.layer());

         //holder->AddNode( child, 1, createPlacement( superlayer ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( superlayer));

         child->SetLineColor( kBlue );
      }
   }
   
   // Fill in DT layer parameters
   auto const & dtLayerGeom = m_geomRecord->slaveGeometry( DTLayerId())->dets();
   for( auto it = dtLayerGeom.begin(),
           end = dtLayerGeom.end(); 
        it != end; ++it )
   {
      if(auto layer = dynamic_cast<const DTLayer*>(*it))
      {

         DTLayerId detid( DetId(layer->geographicalId()));

         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, layer );

         TGeoVolume* holder  = GetDaughter(assembly, "Wheel", detid.wheel());
         holder = GetDaughter(holder, "Station", detid.station());
         holder = GetDaughter(holder, "Sector", detid.sector());
         holder = GetDaughter(holder, "SuperLayer", detid.superlayer());
         holder = GetDaughter(holder, "Layer", detid.layer());

         //         holder->AddNode( child, 1, createPlacement( layer ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement( layer));
         child->SetLineColor( kBlue );
      }
   } 


   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}
//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addCSCGeometry()
{
   if(! m_geomRecord->slaveGeometry( CSCDetId()))
      throw cms::Exception( "FatalError" ) << "Cannnot find CSCGeometry\n";

   
   TGeoVolume *assembly = new TGeoVolumeAssembly("CSC");

   auto const & cscGeom = m_geomRecord->slaveGeometry( CSCDetId())->dets();
   for( auto  it = cscGeom.begin(), itEnd = cscGeom.end(); it != itEnd; ++it )
   {    
      unsigned int rawid = (*it)->geographicalId();
      CSCDetId detId(rawid);
      std::stringstream s;
      s << "CSC" << detId;
      std::string name = s.str();
      
      TGeoVolume* child = 0;

      if( auto chamber = dynamic_cast<const CSCChamber*>(*it))
         child = createVolume( name, chamber );
      else if( auto * layer = dynamic_cast<const CSCLayer*>(*it))
         child = createVolume( name, layer );



      if (child) {
         TGeoVolume* holder  = GetDaughter(assembly, "Endcap", detId.endcap());
         holder = GetDaughter(holder, "Station", detId.station());
         holder = GetDaughter(holder, "Ring", detId.ring());
         holder = GetDaughter(holder, "Chamber", detId.chamber());
      
         child->SetLineColor( kBlue );
         //   holder->AddNode(child, 1,  createPlacement( *it ));
      AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));
      }
   }

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}

//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addGEMGeometry()
{ 
   TGeoVolume *assembly = new TGeoVolumeAssembly("GEM");

   DetId detId( DetId::Muon, MuonSubdetId::GEM );
   const GEMGeometry* gemGeom = (const GEMGeometry*) m_geomRecord->slaveGeometry( detId );
   for( auto it = gemGeom->etaPartitions().begin(),
	   end = gemGeom->etaPartitions().end(); 
        it != end; ++it )
   {
      const GEMEtaPartition* roll = (*it);
      if( roll )
      {
         GEMDetId detid = roll->geographicalId();
         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, roll );

         TGeoVolume* holder  = GetDaughter(assembly, "ROLL Region", detid.region());
         holder = GetDaughter(holder, "Ring", detid.ring());
         holder = GetDaughter(holder, "Station", detid.station()); 
         holder = GetDaughter(holder, "Layer", detid.layer()); 
         holder = GetDaughter(holder, "Chamber", detid.chamber()); 

         AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));

         child->SetLineColor( kYellow );     
      }
   }

   printf("ADD GEM!!!\n");
   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}

//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addRPCGeometry( )
{
   TGeoVolume *assembly = new TGeoVolumeAssembly("RPC");

   DetId detId( DetId::Muon, MuonSubdetId::RPC );
   const RPCGeometry* rpcGeom = (const RPCGeometry*) m_geomRecord->slaveGeometry( detId );
   for( auto it = rpcGeom->rolls().begin(),
           end = rpcGeom->rolls().end(); 
        it != end; ++it )
   {
      RPCRoll const* roll = (*it);
      if( roll )
      {
         RPCDetId detid = roll->geographicalId();
         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, roll );

         TGeoVolume* holder  = GetDaughter(assembly, "ROLL Region", detid.region());
         holder = GetDaughter(holder, "Ring", detid.ring());
         holder = GetDaughter(holder, "Station", detid.station()); 
         holder = GetDaughter(holder, "Sector", detid.sector()); 
         holder = GetDaughter(holder, "Layer", detid.layer()); 
         holder = GetDaughter(holder, "Subsector", detid.subsector()); 

         AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));

         child->SetLineColor( kYellow );     
      }
   }
   /*
  //AMT why no chamber ???
  for( auto it = rpcGeom->chambers().begin(),
           end = rpcGeom->chambers().end(); 
        it != end; ++it )
   {
      RPCChamber const* chamber = (*it);
      if( chamber )
      {
         RPCDetId detid = chamber->geographicalId();
         std::stringstream s;
         s << detid;
         std::string name = s.str();
      
         TGeoVolume* child = createVolume( name, chamber );

         TGeoVolume* holder  = GetDaughter(assembly, "CHANBER Region", detid.region());
         holder = GetDaughter(holder, "Ring", detid.ring());
         holder = GetDaughter(holder, "Station", detid.station()); 
         holder = GetDaughter(holder, "Sector", detid.sector()); 
         holder = GetDaughter(holder, "Layer", detid.layer()); 
         holder = GetDaughter(holder, "Subsector", detid.subsector()); 

         holder->AddNode( child, 1, createPlacement( chamber ));
         child->SetLineColor( kYellow );     
      }
      else printf("NO CHAMBER \n");
   }
   */

   m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
}


//______________________________________________________________________________


//std::map< const CaloCellGeometry::CCGFloat*, TGeoVolume*> m_caloShapeMap;
namespace {
typedef std::map< const float*, TGeoVolume*> CaloVolMap;
CaloVolMap m_caloShapeMap;
}
//______________________________________________________________________________



TGeoVolume* FWTGeoRecoGeometryESProducer::getTruncatedPyramidVolume(const CaloCellGeometry* cell)
{
   TGeoVolume* volume = 0;

   CaloVolMap::iterator volIt =  m_caloShapeMap.find(cell->param());
   if  (volIt == m_caloShapeMap.end()) {
      double points[24];
      const HepGeom::Transform3D idtr;
      std::vector<float> vpar;
      for (int ip =0; ip < 11; ++ip)
         vpar.push_back(cell->param()[ip]);

      std::vector<GlobalPoint> co(8);
      TruncatedPyramid::createCorners(vpar, idtr, co);

      unsigned int index( 0 ); 
      static const int arr[] = {0, 3, 2, 1, 4, 7, 6, 5};
      for( int c = 0; c < 8; ++c )
      {
         points[index] = co[arr[c]].x();
         points[++index] = co[arr[c]].y();
         points[++index] = co[arr[c]].z();
      }  

      TGeoShape* solid = new TGeoArb8(cell->param()[0], points); 


      volume = new TGeoVolume( "TruncatedPyramid" ,solid, m_dummyMedium);
      volume->SetLineColor(kGray);
      m_caloShapeMap[cell->param()]  = volume;
   }
   else {
      volume = volIt->second;
   }

   return volume;
}

//______________________________________________________________________________

TGeoVolume* FWTGeoRecoGeometryESProducer::getIdealZPrismVolume(const CaloCellGeometry* cell)
{
   TGeoVolume* volume = 0;

   return volume;
}

//______________________________________________________________________________

TGeoVolume* FWTGeoRecoGeometryESProducer::getIdealObliquePrismVolume(const CaloCellGeometry* cell)
{
   TGeoVolume* volume = 0;

   // AMT code unfinished ...
   /*
   CaloVolMap::iterator volIt =  m_caloShapeMap.find(cell->param());

   if  (volIt == m_caloShapeMap.end()) {
      printf("FIREWORKS NEW SHAPE BEGIN >>>>>> \n");
     
      double points[24];
      IdealObliquePrism::Pt3DVec co(8);
      IdealObliquePrism::Pt3D ref;
      IdealObliquePrism::localCorners( co, cell->param(), ref);
      
      static const int arr[] = {2, 3, 0, 1, 6, 7, 4,5};
        //static const int arr[] = { 0, 1, 2, 3, 4, 5, 6, 7};

      unsigned int idx( 0 ); 
      for( int c = 0; c < 8; ++c, ++idx )
      {
         points[idx*3]     = co[arr[c]].x();
         points[idx*3 + 1] = co[arr[c]].y();
         points[idx*3 + 2] = co[arr[c]].z();
         printf("fw oblique lc [%d] = %.4f %.4f %.4f\n", c, points[idx*3], points[idx*3+1], points[idx*3+2]);
      }  


       if (cell->param()[2] < 0) {
         printf("negatice height %f\n", cell->param()[2]);
      }

      TGeoShape* solid = new TGeoArb8(TMath::Abs(cell->param()[2]), points); 
      volume = new TGeoVolume( "ObliquePrism" ,solid, m_dummyMedium);
      
       

      //volume =  m_fwGeometry->manager()->MakeBox( "Z prsim", m_dummyMedium, dz, dz, dz );


      volume->SetLineColor(kGray);
      m_caloShapeMap[cell->param()]  = volume;
      CaloCellGeometry::CornersVec const & cv = cell->getCorners();
      printf(" cell ------------------- \n");
      for( int c = 0; c < 8; ++c)
         printf("global cell corners [%d] = %.4f %.4f %.4f\n", c, cv[c].x(),cv[c].y(),cv[c].z() );

      printf("FIREWORKS NEW SHAPE END >>>>>> \n");
   }
   else {
      volume = volIt->second;
   }

   */
   return volume;
}


//______________________________________________________________________________



TGeoVolume* FWTGeoRecoGeometryESProducer::getCalloCellVolume(const CaloCellGeometry* cell)
{
   if (dynamic_cast<const TruncatedPyramid*> (cell)) 
      return getTruncatedPyramidVolume(cell);

   if (dynamic_cast<const IdealZPrism*> (cell))
      return getIdealZPrismVolume(cell);

   if (dynamic_cast<const IdealObliquePrism*> (cell)) 
      return getIdealObliquePrismVolume(cell);

   return 0;
}

//______________________________________________________________________________



void
FWTGeoRecoGeometryESProducer::addEcalCaloGeometry( void )
{
   if (1)
   {
      TGeoVolume *assembly = new TGeoVolumeAssembly("EcalBarrel");
      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Ecal, EcalSubdetector::EcalBarrel);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
         EBDetId detid(*it);
         const CaloCellGeometry* cell = m_caloGeom->getGeometry( *it );
         TGeoVolume* holder  = GetDaughter(assembly, "ism", detid.ism());
         holder->AddNode( getCalloCellVolume(cell), 1, createCaloPlacement(cell));

      }
      m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
   }


   if (1) {
      TGeoVolume *assembly = new TGeoVolumeAssembly("EcalEndcap");
      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Ecal, EcalSubdetector::EcalEndcap);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
         EEDetId detid(*it);
         const CaloCellGeometry* cell = m_caloGeom->getGeometry( *it );

         TGeoVolume* holder  = GetDaughter(assembly, "zside", detid.zside());
         holder  = GetDaughter(holder, "sc", detid.sc());
         holder->AddNode( getCalloCellVolume(cell), 1, createCaloPlacement(cell));
      }
      m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
   }
}

/*

EvePointSet* pointset(Int_t npoints = 512, TEveElement* parent=0)
{
  TEveManager::Create();

  for (int i = 0; i < 8; ++i) {
 
    TEvePointSet* ps = new TEvePointSet(Form("cell %d", i));

    if (i == 0)
      ps->SetNextPoint(180.5827, 15.7989, 0.0000);
    if (i == 1)
      ps->SetNextPoint(181.2725, -0.0000, 0.0000);
    if (i == 2)
      ps->SetNextPoint(181.2725, -0.0000, -15.7906);
    if (i == 3)
      ps->SetNextPoint(180.5827, 15.7989, -15.7906);
    if (i == 4)
      ps->SetNextPoint(287.9751, 25.1945, 0.0000);
    if (i == 5)
      ps->SetNextPoint(289.0751, -0.0000, 0.0000);
    if (i == 6)
      ps->SetNextPoint(289.0751, -0.0000, -25.1813);
    if (i == 7)
      ps->SetNextPoint(287.9751, 25.1945, -25.1813);

    if (i > 3)
      ps->SetMainColor(kOrange);
    else
      ps->SetMainColor(kCyan);

    ps->SetMarkerSize(3);
    ps->SetMarkerStyle(2);
    gEve->AddElement(ps);
  }

  gEve->Redraw3D();
  return ps;
}

void
FWTGeoRecoGeometryESProducer::addHcalCaloGeometry( void )
{

   if (1) {
      TGeoVolume *assembly = new TGeoVolumeAssembly("HcalBarrel");
      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalBarrel);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
         const CaloCellGeometry* cell = m_caloGeom->getGeometry( *it );
         TGeoVolume* v = getCalloCellVolume(cell);
         if (v) assembly->AddNode( getCalloCellVolume(cell), 1, createCaloPlacement(cell));
         if (++mc > 0) break;
      }
      m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
   }
   if (0) {      TGeoVolume *assembly = new TGeoVolumeAssembly("HcalEndcap");
      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalEndcap);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
         const CaloCellGeometry* cell = m_caloGeom->getGeometry( *it );
         TGeoVolume* v = getCalloCellVolume(cell);
         if (v) assembly->AddNode( getCalloCellVolume(cell), 1, createCaloPlacement(cell));

      }
      m_fwGeometry->manager()->GetTopVolume()->AddNode( assembly, 1 );
   }


   printf("SHAPE map size %d n", (int)m_caloShapeMap.size());

}
*/
