#include "Fireworks/Geometry/interface/FWTGeoRecoGeometryESProducer.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
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
#include "TMath.h"

#include "TEveVector.h"
#include "TEveTrans.h"


//std::map< const CaloCellGeometry::CCGFloat*, TGeoVolume*> caloShapeMap;
namespace {
typedef std::map< const float*, TGeoVolume*> CaloVolMap;
}
FWTGeoRecoGeometryESProducer::FWTGeoRecoGeometryESProducer( const edm::ParameterSet& pset )
  : m_dummyMedium(nullptr)
{
  m_tracker = pset.getUntrackedParameter<bool>( "Tracker", true );
  m_muon = pset.getUntrackedParameter<bool>( "Muon", true );
  m_calo = pset.getUntrackedParameter<bool>( "Calo", true );
  
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


}
//______________________________________________________________________________


TGeoVolume* FWTGeoRecoGeometryESProducer::GetDaughter(TGeoVolume* mother, const char* prefix, ERecoDet cidx, int id)
{
   TGeoVolume* res = nullptr;
   if (mother->GetNdaughters()) { 
      TGeoNode* n = mother->FindNode(Form("%s_%d_1", prefix, id));
      if ( n ) res = n->GetVolume();
   }

   if (!res) {
      res = new TGeoVolumeAssembly( Form("%s_%d", prefix, id ));
      res->SetMedium(GetMedium(cidx));
      mother->AddNode(res, 1);
   }

   return res;
}

TGeoVolume* FWTGeoRecoGeometryESProducer::GetDaughter(TGeoVolume* mother, const char* prefix, ERecoDet cidx)
{
   TGeoVolume* res = nullptr;
   if (mother->GetNdaughters()) { 
      TGeoNode* n = mother->FindNode(Form("%s_1",prefix));
      if ( n ) res = n->GetVolume();
   }

   if (!res) {
      //      printf("GetDau... new holder %s for mother %s \n", mother->GetName(), prefix);
      res = new TGeoVolumeAssembly(prefix);
      res->SetMedium(GetMedium(cidx));
      mother->AddNode(res, 1);
   }

   return res;
}

TGeoVolume* FWTGeoRecoGeometryESProducer::GetTopHolder(const char* prefix, ERecoDet cidx)
{
   //   printf("GetTopHolder res = %s \n", prefix);
   TGeoVolume* res =  GetDaughter(gGeoManager->GetTopVolume(), prefix, cidx);
   return res;
}

namespace {

enum GMCol { Green = 4, 
             Blue0 = 13, Blue1 = 24, Blue2 = 6,
             Yellow0 = 3, Yellow1 = 16,
             Pink = 10,  
             Red = 29, Orange0 = 79, Orange1 = 14,
             Magenta = 8,
             Gray = 12
};

}


TGeoMedium*
FWTGeoRecoGeometryESProducer::GetMedium(ERecoDet det)
{
   std::map<ERecoDet, TGeoMedium*>::iterator it = m_recoMedium.find(det);
   if (it != m_recoMedium.end())
      return it->second;

   std::string name;
   int color;


   switch (det)
   {
      // TRACKER
      case kSiPixel:
         name = "SiPixel";
         color = GMCol::Green;
         break;

      case kSiStrip:
         name = "SiStrip";
         color = GMCol::Gray;
         break;
         // MUON
      case kMuonDT:
         name = "MuonDT";
         color = GMCol::Blue2;
         break;

      case kMuonRPC:
         name = "MuonRPC";
         color = GMCol::Red;
         break;

      case kMuonGEM:
         name = "MuonGEM";
         color = GMCol::Yellow1;
         break;

      case kMuonCSC:
         name = "MuonCSC";
         color = GMCol::Gray;
         break;

      case kMuonME0:
         name = "MuonME0";
         color = GMCol::Yellow0;
         break;

         // CALO
      case kECal:
         name = "ECal";
         color = GMCol::Blue2;
         break;
      case kHCal:
         name = "HCal";    
         color = GMCol::Orange1;
         break;
      case kCaloTower:
         name = "CaloTower";    
         color = GMCol::Green;
         break;
      case kHGCE:
         name = "HGCEE";
         color = GMCol::Blue2;
         break;
      case kHGCH:
         name = "HGCEH";
         color = GMCol::Blue1;
         break;
      default:
         printf("invalid medium id \n");
         return m_dummyMedium;
   }

   TGeoMaterial* mat = new TGeoMaterial(name.c_str(), 0, 0, 0);
   mat->SetZ(color);
   m_recoMedium[det] = new TGeoMedium(name.c_str(), 0, mat);
   mat->SetFillStyle(3000); // tansparency 3000-3100
   mat->SetDensity(1); // disable override of transparency in TGeoManager::DefaultColors()

   return m_recoMedium[det];
}


//______________________________________________________________________________



std::unique_ptr<FWTGeoRecoGeometry> 
FWTGeoRecoGeometryESProducer::produce( const FWTGeoRecoGeometryRecord& record )
{
   using namespace edm;

   auto fwTGeoRecoGeometry = std::make_unique<FWTGeoRecoGeometry>();
  
   if( m_calo ) {
     edm::ESHandle<CaloGeometry> caloH;
     record.getRecord<CaloGeometryRecord>().get(caloH);
     m_caloGeom = caloH.product();
   }

   TGeoManager* geom = new TGeoManager( "cmsGeo", "CMS Detector" );
   if( nullptr == gGeoIdentity )
   {
      gGeoIdentity = new TGeoIdentity( "Identity" );
   }

   fwTGeoRecoGeometry->manager( geom );
  
   // Default material is Vacuum
   TGeoMaterial *vacuum = new TGeoMaterial( "Vacuum", 0 ,0 ,0 );
   m_dummyMedium = new TGeoMedium( "reco", 0, vacuum);

   TGeoVolume *top = geom->MakeBox( "CMS", m_dummyMedium, 270., 270., 120. );
  
   if( nullptr == top )
   {
      return std::unique_ptr<FWTGeoRecoGeometry>();
   }
   geom->SetTopVolume( top );
   // ROOT chokes unless colors are assigned
   top->SetVisibility( kFALSE );
   top->SetLineColor( kBlue );

   if( m_tracker || m_muon )
   {
     record.getRecord<GlobalTrackingGeometryRecord>().get( m_geomRecord );
   }
   
   if( m_tracker )
   {
     DetId detId( DetId::Tracker, 0 );
     m_trackerGeom = (const TrackerGeometry*) m_geomRecord->slaveGeometry( detId );
     
     edm::ESHandle<TrackerTopology> trackerTopologyHandle;
     record.getRecord<TrackerTopologyRcd>().get( trackerTopologyHandle );
     m_trackerTopology = trackerTopologyHandle.product();
     
     addPixelBarrelGeometry();
     addPixelForwardGeometry();

     addTIBGeometry();
     addTIDGeometry();
     addTOBGeometry();
     addTECGeometry();
   }
   
   if( m_muon )
   {
     addDTGeometry();
     addCSCGeometry();
     addRPCGeometry();
     addME0Geometry();
     addGEMGeometry();
   }

   if( m_calo )
   {  
     addEcalCaloGeometry();   
     addHcalCaloGeometryBarrel();
     addHcalCaloGeometryEndcap();
     addHcalCaloGeometryOuter();
     addHcalCaloGeometryForward();
     addCaloTowerGeometry();
   }
   
   geom->CloseGeometry();

   geom->DefaultColors();
   // printf("==== geo manager NNodes = %d \n", geom->GetNNodes());
   geom->CloseGeometry();

   return fwTGeoRecoGeometry;
}

/** Create TGeo shape for GeomDet */
TGeoShape* 
FWTGeoRecoGeometryESProducer::createShape( const GeomDet *det )
{
  TGeoShape* shape = nullptr;

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
    if( nullptr == shape )
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
  if( dynamic_cast<const RectangularPlaneBounds *> (b) != nullptr )
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
    if( nullptr == shape )
    {
      shape = new TGeoBBox( name.c_str(), width / 2., length / 2., thickness / 2. ); // dx, dy, dz

      m_nameToShape[name] = shape;
    }
  }
  
  return shape;
}

/** Create TGeo volume for GeomDet */
TGeoVolume* 
FWTGeoRecoGeometryESProducer::createVolume( const std::string& name, const GeomDet *det, ERecoDet mid )
{
   TGeoShape* solid = createShape( det );

   std::map<TGeoShape*, TGeoVolume*>::iterator vIt = m_shapeToVolume.find(solid);
   if (vIt != m_shapeToVolume.end()) return  vIt->second;
   

   TGeoVolume* volume = new TGeoVolume( name.c_str(),solid, GetMedium(mid));

   m_shapeToVolume[solid] = volume;

   return volume;
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
  TGeoVolume* tv = GetTopHolder( "SiPixel", kSiPixel );
  TGeoVolume *assembly = GetDaughter( tv, "PXB", kSiPixel );

  for( auto it : m_trackerGeom->detsPXB()) {
    DetId detid = it->geographicalId();
    unsigned int layer = m_trackerTopology->pxbLayer( detid );
    unsigned int module = m_trackerTopology->pxbModule( detid );
    unsigned int ladder = m_trackerTopology->pxbLadder( detid );
    
    std::string name = Form( "PXB Ly:%d, Md:%d Ld:%d ", layer, module, ladder );
    TGeoVolume* child = createVolume( name, it, kSiPixel );
    
    TGeoVolume* holder = GetDaughter( assembly, "Layer", kSiPixel, layer );
    holder = GetDaughter( holder, "Module", kSiPixel, module );
    
    AddLeafNode( holder, child, name.c_str(), createPlacement( it ));
  }
}

void
FWTGeoRecoGeometryESProducer::addPixelForwardGeometry()
{
  TGeoVolume* tv = GetTopHolder( "SiPixel", kSiPixel );
  TGeoVolume* assembly = GetDaughter( tv, "PXF", kSiPixel );

  for( auto it : m_trackerGeom->detsPXF()) {
    DetId detid = it->geographicalId();
    unsigned int disk = m_trackerTopology->pxfDisk( detid );
    unsigned int blade = m_trackerTopology->pxfBlade( detid );
    unsigned int panel = m_trackerTopology->pxfPanel( detid );
    unsigned int side = m_trackerTopology->side( detid );
    
    std::string name = Form( "PXF D:%d, B:%d, P:%d, S:%d", disk, blade, panel, side );
    TGeoVolume* child = createVolume( name, it, kSiPixel );

    TGeoVolume* holder = GetDaughter( assembly, "Side", kSiPixel, side );
    holder = GetDaughter( holder, "Disk", kSiPixel, disk );
    holder = GetDaughter( holder, "Blade", kSiPixel, blade );
    holder = GetDaughter( holder, "Panel", kSiPixel, panel );
   
    AddLeafNode( holder, child, name.c_str(), createPlacement( it ));
  }
}

void
FWTGeoRecoGeometryESProducer::addTIBGeometry()
{
  TGeoVolume* tv = GetTopHolder( "SiStrip", kSiStrip );
  TGeoVolume *assembly = GetDaughter( tv, "TIB", kSiStrip );

  for( auto it : m_trackerGeom->detsTIB()) {
    DetId detid = it->geographicalId();
    unsigned int module = m_trackerTopology->tibModule( detid );
    unsigned int order = m_trackerTopology->tibOrder( detid );
    unsigned int side = m_trackerTopology->tibSide( detid );

    std::string name = m_trackerTopology->print(detid);
    
    TGeoVolume* child = createVolume( name, it, kSiStrip );
    TGeoVolume* holder = GetDaughter( assembly, "Module", kSiStrip, module );
    holder = GetDaughter( holder, "Order", kSiStrip, order );
    holder = GetDaughter( holder, "Side", kSiStrip, side );
    AddLeafNode( holder, child, name.c_str(), createPlacement( it ));
  }
}

void
FWTGeoRecoGeometryESProducer::addTIDGeometry()
{
  TGeoVolume* tv = GetTopHolder( "SiStrip", kSiStrip );
  TGeoVolume* assembly = GetDaughter( tv, "TID", kSiStrip );

  for( auto it : m_trackerGeom->detsTID()) {
    DetId detid = it->geographicalId();
    unsigned int side = m_trackerTopology->tidSide( detid );
    unsigned int wheel = m_trackerTopology->tidWheel( detid );
    unsigned int ring = m_trackerTopology->tidRing( detid );

    std::string name = m_trackerTopology->print(detid);

    TGeoVolume* child = createVolume( name, it, kSiStrip );
    TGeoVolume* holder = GetDaughter( assembly, "Side", kSiStrip, side );
    holder = GetDaughter( holder, "Wheel", kSiStrip, wheel );
    holder = GetDaughter( holder, "Ring", kSiStrip, ring );
    AddLeafNode( holder, child, name.c_str(),  createPlacement( it ));
  }
}

void
FWTGeoRecoGeometryESProducer::addTOBGeometry()
{
  TGeoVolume* tv = GetTopHolder( "SiStrip", kSiStrip );
  TGeoVolume* assembly = GetDaughter( tv, "TOB",  kSiStrip );

  for( auto it : m_trackerGeom->detsTOB()) {
    DetId detid = it->geographicalId();
    unsigned int rod = m_trackerTopology->tobRod( detid );
    unsigned int side = m_trackerTopology->tobSide( detid );
    unsigned int module = m_trackerTopology->tobModule( detid );

    std::string name = m_trackerTopology->print(detid);

    TGeoVolume* child = createVolume( name, it, kSiStrip );
    TGeoVolume* holder = GetDaughter( assembly, "Rod", kSiStrip, rod );
    holder = GetDaughter( holder, "Side", kSiStrip, side );
    holder = GetDaughter( holder, "Module", kSiStrip, module );
    AddLeafNode( holder, child, name.c_str(), createPlacement( it ));
  }
}

void
FWTGeoRecoGeometryESProducer::addTECGeometry()
{
  TGeoVolume* tv = GetTopHolder( "SiStrip", kSiStrip );
  TGeoVolume* assembly = GetDaughter( tv, "TEC", kSiStrip );

  for( auto it : m_trackerGeom->detsTEC()) {
    DetId detid = it->geographicalId();
    unsigned int order = m_trackerTopology->tecOrder( detid );
    unsigned int ring = m_trackerTopology->tecRing( detid );
    unsigned int module = m_trackerTopology->tecModule( detid );

    std::string name = m_trackerTopology->print(detid);

    TGeoVolume* child = createVolume( name, it, kSiStrip );

    TGeoVolume* holder = GetDaughter( assembly, "Order", kSiStrip, order );
    holder = GetDaughter( holder, "Ring", kSiStrip, ring );
    holder = GetDaughter( holder, "Module", kSiStrip, module );
    AddLeafNode( holder, child, name.c_str(), createPlacement( it ));
  }
}

//==============================================================================
//==============================================================================
//=================================== MUON =====================================
//==============================================================================



void
FWTGeoRecoGeometryESProducer::addDTGeometry(  )
{
   TGeoVolume* tv =  GetTopHolder("Muon", kMuonRPC);
   TGeoVolume *assemblyTop = GetDaughter(tv, "DT", kMuonDT);

   //
   // DT chambers geometry
   //
   {
      TGeoVolume *assembly = GetDaughter(assemblyTop, "DTChamber", kMuonDT);
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
      
            TGeoVolume* child = createVolume( name, chamber, kMuonDT );
            TGeoVolume* holder  = GetDaughter(assembly, "Wheel", kMuonDT, detid.wheel());
            holder = GetDaughter(holder, "Station", kMuonDT, detid.station());
            holder = GetDaughter(holder, "Sector", kMuonDT, detid.sector());
   
            AddLeafNode(holder, child, name.c_str(),  createPlacement( chamber));
         }
      }
   }

   // Fill in DT super layer parameters
   {
      TGeoVolume *assembly = GetDaughter(assemblyTop, "DTSuperLayer", kMuonDT);
      auto const & dtSuperLayerGeom = m_geomRecord->slaveGeometry( DTSuperLayerId())->dets();
      for( auto it = dtSuperLayerGeom.begin(),
              end = dtSuperLayerGeom.end(); 
           it != end; ++it )
      {
         if( auto * superlayer = dynamic_cast<const DTSuperLayer*>(*it))
         {
            DTSuperLayerId detid( DetId(superlayer->geographicalId()));
            std::stringstream s;
            s << detid;
            std::string name = s.str();
      
            TGeoVolume* child = createVolume( name, superlayer, kMuonDT );

            TGeoVolume* holder  = GetDaughter(assembly, "Wheel", kMuonDT, detid.wheel());
            holder = GetDaughter(holder, "Station", kMuonDT, detid.station());
            holder = GetDaughter(holder, "Sector", kMuonDT, detid.sector());
            holder = GetDaughter(holder, "SuperLayer", kMuonDT, detid.superlayer());
            AddLeafNode(holder, child, name.c_str(),  createPlacement( superlayer));
         }
      }
   }
   // Fill in DT layer parameters
   {
      TGeoVolume *assembly = GetDaughter(assemblyTop, "DTLayer", kMuonDT);
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
      
            TGeoVolume* child = createVolume( name, layer, kMuonDT );

            TGeoVolume* holder  = GetDaughter(assembly, "Wheel", kMuonDT, detid.wheel());
            holder = GetDaughter(holder, "Station", kMuonDT, detid.station());
            holder = GetDaughter(holder, "Sector", kMuonDT, detid.sector());
            holder = GetDaughter(holder, "SuperLayer", kMuonDT, detid.superlayer());
            holder = GetDaughter(holder, "Layer", kMuonDT, detid.layer());
            AddLeafNode(holder, child, name.c_str(),  createPlacement( layer));
         }
      } 
   }
}
//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addCSCGeometry()
{
   if(! m_geomRecord->slaveGeometry( CSCDetId()))
      throw cms::Exception( "FatalError" ) << "Cannnot find CSCGeometry\n";

   
   TGeoVolume* tv =  GetTopHolder("Muon", kMuonRPC);
   TGeoVolume *assembly = GetDaughter(tv, "CSC", kMuonCSC);

   auto const & cscGeom = m_geomRecord->slaveGeometry( CSCDetId())->dets();
   for( auto  it = cscGeom.begin(), itEnd = cscGeom.end(); it != itEnd; ++it )
   {    
      unsigned int rawid = (*it)->geographicalId();
      CSCDetId detId(rawid);
      std::stringstream s;
      s << "CSC" << detId;
      std::string name = s.str();
      
      TGeoVolume* child = nullptr;

      if( auto chamber = dynamic_cast<const CSCChamber*>(*it))
         child = createVolume( name, chamber, kMuonCSC );
      else if( auto * layer = dynamic_cast<const CSCLayer*>(*it))
         child = createVolume( name, layer, kMuonCSC );



      if (child) {
         TGeoVolume* holder  = GetDaughter(assembly, "Endcap", kMuonCSC, detId.endcap());
         holder = GetDaughter(holder, "Station", kMuonCSC, detId.station());
         holder = GetDaughter(holder, "Ring", kMuonCSC, detId.ring());
         holder = GetDaughter(holder, "Chamber", kMuonCSC , detId.chamber());
      
         //   holder->AddNode(child, 1,  createPlacement( *it ));
         AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));
      }
   }

}

//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addGEMGeometry()
{ 
   try {
      DetId detId( DetId::Muon, MuonSubdetId::GEM );
      const GEMGeometry* gemGeom = (const GEMGeometry*) m_geomRecord->slaveGeometry( detId );

      TGeoVolume* tv =  GetTopHolder("Muon", kMuonRPC);
      TGeoVolume *assemblyTop = GetDaughter(tv, "GEM", kMuonGEM);
      
      {
	TGeoVolume *assembly = GetDaughter(assemblyTop, "GEMSuperChambers", kMuonGEM);      
	for( auto it = gemGeom->superChambers().begin(),
	       end = gemGeom->superChambers().end(); 
	     it != end; ++it )
	  {
	    const GEMSuperChamber* sc = (*it);
	    if( sc )
	      {
		GEMDetId detid = sc->geographicalId();
		std::stringstream s;
		s << detid;
		std::string name = s.str();
      
		TGeoVolume* child = createVolume( name, sc, kMuonGEM );

		TGeoVolume* holder  = GetDaughter(assembly, "SuperChamber Region", kMuonGEM , detid.region());
		holder = GetDaughter(holder, "Ring", kMuonGEM , detid.ring());
		holder = GetDaughter(holder, "Station", kMuonGEM , detid.station()); 
		holder = GetDaughter(holder, "Chamber", kMuonGEM , detid.chamber()); 

		AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));
	      }
	  }
      }
      
      {
	TGeoVolume *assembly = GetDaughter(assemblyTop, "GEMetaPartitions", kMuonGEM);      
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
      
		TGeoVolume* child = createVolume( name, roll, kMuonGEM );

		TGeoVolume* holder  = GetDaughter(assembly, "ROLL Region", kMuonGEM , detid.region());
		holder = GetDaughter(holder, "Ring", kMuonGEM , detid.ring());
		holder = GetDaughter(holder, "Station", kMuonGEM , detid.station()); 
		holder = GetDaughter(holder, "Layer", kMuonGEM , detid.layer()); 
		holder = GetDaughter(holder, "Chamber", kMuonGEM , detid.chamber()); 

		AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));
	      }
	  }
      }
   }catch (cms::Exception &exception) {
     edm::LogInfo("FWRecoGeometry") << "failed to produce GEM geometry " << exception.what() << std::endl;
     
   }
}

//______________________________________________________________________________


void
FWTGeoRecoGeometryESProducer::addRPCGeometry( )
{
   TGeoVolume* tv =  GetTopHolder("Muon", kMuonRPC);
   TGeoVolume *assembly = GetDaughter(tv, "RPC", kMuonRPC);

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
      
         TGeoVolume* child = createVolume( name, roll, kMuonRPC );

         TGeoVolume* holder  = GetDaughter(assembly, "ROLL Region", kMuonRPC, detid.region());
         holder = GetDaughter(holder, "Ring", kMuonRPC, detid.ring());
         holder = GetDaughter(holder, "Station", kMuonRPC, detid.station()); 
         holder = GetDaughter(holder, "Sector", kMuonRPC, detid.sector()); 
         holder = GetDaughter(holder, "Layer", kMuonRPC, detid.layer()); 
         holder = GetDaughter(holder, "Subsector", kMuonRPC, detid.subsector()); 
 
         AddLeafNode(holder, child, name.c_str(),  createPlacement(*it));
      }
   };
}

void
FWTGeoRecoGeometryESProducer::addME0Geometry( )
{
  TGeoVolume* tv =  GetTopHolder("Muon", kMuonME0);
  TGeoVolume *assembly = GetDaughter(tv, "ME0", kMuonME0);

  DetId detId( DetId::Muon, 5 );
  try 
    {
      const ME0Geometry* me0Geom = (const ME0Geometry*) m_geomRecord->slaveGeometry( detId );

      for(auto roll : me0Geom->etaPartitions())
	{ 
	  if( roll )
	    {
	      unsigned int rawid = roll->geographicalId().rawId();
	      //std::cout << "AMT FWTTTTRecoGeometryES\n" << rawid ;
                        
	      ME0DetId detid(rawid);
	      std::stringstream s;
	      s << detid;
	      std::string name = s.str();
	      TGeoVolume* child = createVolume( name, roll, kMuonME0 );

	      TGeoVolume* holder  = GetDaughter(assembly, "Region", kMuonME0, detid.region());
	      holder = GetDaughter(holder, "Layer", kMuonME0, detid.layer()); 
	      holder = GetDaughter(holder, "Chamber", kMuonME0, detid.chamber()); 
	      AddLeafNode(holder, child, name.c_str(),  createPlacement(roll));


	    }
	}
    }
  catch( cms::Exception &exception )
    {
      edm::LogInfo("FWRecoGeometry") << "failed to produce ME0 geometry " << exception.what() << std::endl;
    }
}


//==============================================================================
//================================= CALO =======================================
//==============================================================================


void
FWTGeoRecoGeometryESProducer::addHcalCaloGeometryBarrel( void )
{
   TGeoVolume* tv =  GetTopHolder("HCal", kHCal); 
   TGeoVolume *assembly = GetDaughter(tv, "HCalBarrel", kHCal);
   
   std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalBarrel);

   CaloVolMap caloShapeMapP;
   CaloVolMap caloShapeMapN;
   for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
   {
     //HcalDetId detid = HcalDetId(it->rawId());
     HcalDetId detid(*it);
     const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
     const IdealObliquePrism* cell = dynamic_cast<const IdealObliquePrism*> (cellb);
   
     if (!cell) { printf ("HB not oblique !!!\n"); continue; }
  
     TGeoVolume* volume = nullptr;
     CaloVolMap& caloShapeMap = (cell->etaPos() > 0) ? caloShapeMapP : caloShapeMapN;
     CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
     if  (volIt == caloShapeMap.end()) 
        {
         // printf("FIREWORKS NEW SHAPE BEGIN eta = %f etaPos = %f, phiPos %f >>>>>> \n", cell->eta(), cell->etaPos(), cell->phiPos());
         IdealObliquePrism::Pt3DVec lc(8);
         IdealObliquePrism::Pt3D ref;
         IdealObliquePrism::localCorners( lc, cell->param(), ref );
         HepGeom::Vector3D<float> lCenter;
         for( int c = 0; c < 8; ++c)
     	   lCenter += lc[c];
         lCenter *= 0.125;

         static const int arr[] = { 1, 0, 3, 2,  5, 4, 7, 6 };
         double points[16];
         for (int c = 0; c < 8; ++c) {
     	   if (cell->etaPos() > 0 )
     	     points[ c*2 + 0 ] = -(lc[arr[c]].z() - lCenter.z());
     	   else
     	     points[ c*2 + 0 ] = (lc[arr[c]].z() - lCenter.z()); 
     	   points[ c*2 + 1 ] =  (lc[arr[c]].y() - lCenter.y());
            // printf("AMT xy[%d] <=>[%d] = (%.4f, %.4f) \n", arr[c], c, points[c*2],  points[c*2+1]);
         }

         float dz = (lc[4].x() -lc[0].x()) * 0.5;
         TGeoShape* solid = new TGeoArb8(dz, &points[0]);
         volume = new TGeoVolume("hcal oblique prism", solid, GetMedium(kHCal));
         caloShapeMap[cell->param()] = volume;
       }
     else {
       volume = volIt->second;
     }      

      HepGeom::Vector3D<float> gCenter;
      CaloCellGeometry::CornersVec const & gc = cell->getCorners();
      for (int c = 0; c < 8; ++c)
	gCenter += HepGeom::Vector3D<float>(gc[c].x(), gc[c].y(), gc[c].z());
      gCenter *= 0.125;

      TGeoTranslation gtr(gCenter.x(), gCenter.y(), gCenter.z());
      TGeoRotation rot; 
      rot.RotateY(90);
     
      TGeoRotation rotPhi;
      rotPhi.SetAngles(0, -cell->phiPos()*TMath::RadToDeg(), 0);
      rot.MultiplyBy(&rotPhi);    

      TGeoVolume* holder  = GetDaughter(assembly, "side", kHCal, detid.zside());
      holder = GetDaughter(holder, "ieta", kHCal, detid.ieta());
      std::stringstream nname;
      nname << detid;
      AddLeafNode(holder, volume, nname.str().c_str(), new TGeoCombiTrans(gtr, rot));
   }

   //  printf("HB map size P = %lu , N = %lu", caloShapeMapP.size(),caloShapeMapN.size() );
}
//______________________________________________________________________________

void
FWTGeoRecoGeometryESProducer::addHcalCaloGeometryEndcap( void )
{

   CaloVolMap caloShapeMapP;
   CaloVolMap caloShapeMapN;

   TGeoVolume* tv =  GetTopHolder("HCal", kHCal); 
   TGeoVolume *assembly = GetDaughter(tv, "HCalEndcap", kHCal);

   std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalEndcap);

   for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
   {
      HcalDetId detid = HcalDetId(it->rawId());
      const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
      const IdealObliquePrism* cell = dynamic_cast<const IdealObliquePrism*> (cellb);
   
      if (!cell) { printf ("EC not oblique \n"); continue; }

      TGeoVolume* volume = nullptr;
      CaloVolMap& caloShapeMap = (cell->etaPos() > 0) ? caloShapeMapP : caloShapeMapN;
      CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
      if  ( volIt == caloShapeMap.end()) 
      {
         IdealObliquePrism::Pt3DVec lc(8);
         IdealObliquePrism::Pt3D ref;
         IdealObliquePrism::localCorners( lc, cell->param(), ref);
         HepGeom::Vector3D<float> lCenter; 
         for( int c = 0; c < 8; ++c)
            lCenter += lc[c];
         lCenter *= 0.125;

         //for( int c = 0; c < 8; ++c)
         //   printf("lc.push_back(TEveVector(%.4f, %.4f, %.4f));\n", lc[c].x(), lc[c].y(), lc[c].z() );


         static const int arrP[] = { 3, 2, 1, 0, 7, 6, 5, 4 };
         static const int arrN[] = {  7, 6, 5, 4 ,3, 2, 1, 0};
         const int* arr = (detid.ieta() > 0) ?  &arrP[0] : &arrN[0];

         double points[16];
         for (int c = 0; c < 8; ++c) {
            points[ c*2 + 0 ] = lc[arr[c]].x() - lCenter.x(); 
            points[ c*2 + 1 ] = lc[arr[c]].y() - lCenter.y();
         }

         float dz = (lc[4].z() -lc[0].z()) * 0.5;
         TGeoShape* solid = new TGeoArb8(dz, &points[0]);
         volume = new TGeoVolume("ecal oblique prism", solid, GetMedium(kHCal));
         caloShapeMap[cell->param()] = volume;
      }
      else {

         volume = volIt->second;

      }      

      HepGeom::Vector3D<float> gCenter;
      CaloCellGeometry::CornersVec const & gc = cell->getCorners();
      for (int c = 0; c < 8; ++c) {
         gCenter += HepGeom::Vector3D<float>(gc[c].x(), gc[c].y(), gc[c].z());
         //  printf("gc.push_back(TEveVector(%.4f, %.4f, %.4f));\n", gc[c].x(), gc[c].y(),gc[c].z() );
      }
      gCenter *= 0.125;

      TGeoTranslation gtr(gCenter.x(), gCenter.y(), gCenter.z());
      TGeoRotation rot;
      rot.SetAngles(cell->phiPos()*TMath::RadToDeg(), 0, 0);

      TGeoVolume* holder  = GetDaughter(assembly, "side", kHCal, detid.zside());
      holder = GetDaughter(holder, "ieta", kHCal, detid.ieta());
      std::stringstream nname;
      nname << detid;
      AddLeafNode(holder, volume, nname.str().c_str(), new TGeoCombiTrans(gtr, rot));
   }

   //   printf("HE map size P = %lu , N = %lu", caloShapeMapP.size(),caloShapeMapN.size() );
}

void
FWTGeoRecoGeometryESProducer::addHcalCaloGeometryOuter()
{
  CaloVolMap caloShapeMapP;
  CaloVolMap caloShapeMapN;

  TGeoVolume* tv =  GetTopHolder("HCal", kHCal); 
  TGeoVolume *assembly = GetDaughter(tv, "HCalOuter", kHCal);

  std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalOuter);

  for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
  {
    HcalDetId detid = HcalDetId(it->rawId());
    const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
    const IdealObliquePrism* cell = dynamic_cast<const IdealObliquePrism*> (cellb);
   
    if (!cell) { printf ("EC not oblique \n"); continue; }

    TGeoVolume* volume = nullptr;
    CaloVolMap& caloShapeMap = (cell->etaPos() > 0) ? caloShapeMapP : caloShapeMapN;
    CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
    if  ( volIt == caloShapeMap.end()) 
    {
      IdealObliquePrism::Pt3DVec lc(8);
      IdealObliquePrism::Pt3D ref;
      IdealObliquePrism::localCorners( lc, cell->param(), ref);
      HepGeom::Vector3D<float> lCenter; 
      for( int c = 0; c < 8; ++c)
	lCenter += lc[c];
      lCenter *= 0.125;
      static const int arrP[] = { 3, 2, 1, 0, 7, 6, 5, 4 };
      static const int arrN[] = {  7, 6, 5, 4 ,3, 2, 1, 0};
      const int* arr = (detid.ieta() > 0) ?  &arrP[0] : &arrN[0];
      
      double points[16];
      for (int c = 0; c < 8; ++c) {
	points[ c*2 + 0 ] = lc[arr[c]].x() - lCenter.x(); 
	points[ c*2 + 1 ] = lc[arr[c]].y() - lCenter.y();
      }

      float dz = (lc[4].z() -lc[0].z()) * 0.5;
      TGeoShape* solid = new TGeoArb8(dz, &points[0]);
      volume = new TGeoVolume("ecal oblique prism", solid, GetMedium(kHCal));
      caloShapeMap[cell->param()] = volume;
    }
    else {
      volume = volIt->second;
    }      
    HepGeom::Vector3D<float> gCenter;
    CaloCellGeometry::CornersVec const & gc = cell->getCorners();
    for (int c = 0; c < 8; ++c) {
      gCenter += HepGeom::Vector3D<float>(gc[c].x(), gc[c].y(), gc[c].z());
    }
    gCenter *= 0.125;
    
    TGeoTranslation gtr(gCenter.x(), gCenter.y(), gCenter.z());
    TGeoRotation rot;
    rot.SetAngles(cell->phiPos()*TMath::RadToDeg(), 0, 0);
    
    TGeoVolume* holder  = GetDaughter(assembly, "side", kHCal, detid.zside());
    holder = GetDaughter(holder, "ieta", kHCal, detid.ieta());
    std::stringstream nname;
    nname << detid;
    AddLeafNode(holder, volume, nname.str().c_str(), new TGeoCombiTrans(gtr, rot));
  }
}

void
FWTGeoRecoGeometryESProducer::addHcalCaloGeometryForward()
{
  CaloVolMap caloShapeMapP;
  CaloVolMap caloShapeMapN;

  TGeoVolume* tv =  GetTopHolder("HCal", kHCal); 
  TGeoVolume *assembly = GetDaughter(tv, "HCalForward", kHCal);

  std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Hcal, HcalSubdetector::HcalForward);

  for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
  {
    HcalDetId detid = HcalDetId(it->rawId());
    const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
    const IdealZPrism* cell = dynamic_cast<const IdealZPrism*> (cellb);
    
    if (!cell) { printf ("EC not Z prism \n"); continue; }
    
    TGeoVolume* volume = nullptr;
    CaloVolMap& caloShapeMap = (cell->etaPos() > 0) ? caloShapeMapP : caloShapeMapN;
    CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
    if  ( volIt == caloShapeMap.end()) 
    {
      IdealZPrism::Pt3DVec lc(8);
      IdealZPrism::Pt3D ref;
      IdealZPrism::localCorners( lc, cell->param(), ref);
      HepGeom::Vector3D<float> lCenter; 
      for( int c = 0; c < 8; ++c)
	lCenter += lc[c];
      lCenter *= 0.125;
      static const int arrP[] = { 3, 2, 1, 0, 7, 6, 5, 4 };
      static const int arrN[] = {  7, 6, 5, 4 ,3, 2, 1, 0};
      const int* arr = (detid.ieta() > 0) ?  &arrP[0] : &arrN[0];

      double points[16];
      for (int c = 0; c < 8; ++c) {
	points[ c*2 + 0 ] = lc[arr[c]].x() - lCenter.x(); 
	points[ c*2 + 1 ] = lc[arr[c]].y() - lCenter.y();
      }

      float dz = (lc[4].z() -lc[0].z()) * 0.5;
      TGeoShape* solid = new TGeoArb8(dz, &points[0]);
      volume = new TGeoVolume("ecal oblique prism", solid, GetMedium(kHCal));
      caloShapeMap[cell->param()] = volume;
    }
    else {
      volume = volIt->second;
    }      
    HepGeom::Vector3D<float> gCenter;
    CaloCellGeometry::CornersVec const & gc = cell->getCorners();
    for (int c = 0; c < 8; ++c) {
      gCenter += HepGeom::Vector3D<float>(gc[c].x(), gc[c].y(), gc[c].z());
    }
    gCenter *= 0.125;

    TGeoTranslation gtr(gCenter.x(), gCenter.y(), gCenter.z());
    TGeoRotation rot;
    rot.SetAngles(cell->phiPos()*TMath::RadToDeg(), 0, 0);
    
    TGeoVolume* holder  = GetDaughter(assembly, "side", kHCal, detid.zside());
    holder = GetDaughter(holder, "ieta", kHCal, detid.ieta());
    std::stringstream nname;
    nname << detid;
    AddLeafNode(holder, volume, nname.str().c_str(), new TGeoCombiTrans(gtr, rot));
  }
}

void
FWTGeoRecoGeometryESProducer::addCaloTowerGeometry()
{
  CaloVolMap caloShapeMapP;
  CaloVolMap caloShapeMapN;

  TGeoVolume* tv =  GetTopHolder("CaloTower", kCaloTower); 
  TGeoVolume *assembly = GetDaughter(tv, "CaloTower", kCaloTower);
   
  std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Calo, CaloTowerDetId::SubdetId);
  for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
  {
    CaloTowerDetId detid = CaloTowerDetId(it->rawId());
    const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
    const IdealObliquePrism* cell = dynamic_cast<const IdealObliquePrism*> (cellb);
    if (!cell) { printf ("EC not oblique \n"); continue; }
      TGeoVolume* volume = nullptr;
      CaloVolMap& caloShapeMap = (cell->etaPos() > 0) ? caloShapeMapP : caloShapeMapN;
      CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
      if  ( volIt == caloShapeMap.end()) 
      {
         IdealObliquePrism::Pt3DVec lc(8);
         IdealObliquePrism::Pt3D ref;
         IdealObliquePrism::localCorners( lc, cell->param(), ref);
         HepGeom::Vector3D<float> lCenter; 
         for( int c = 0; c < 8; ++c)
            lCenter += lc[c];
         lCenter *= 0.125;

         static const int arrP[] = { 3, 2, 1, 0, 7, 6, 5, 4 };
         static const int arrN[] = {  7, 6, 5, 4 ,3, 2, 1, 0};
         const int* arr = (detid.ieta() > 0) ?  &arrP[0] : &arrN[0];

         double points[16];
         for (int c = 0; c < 8; ++c) {
            points[ c*2 + 0 ] = lc[arr[c]].x() - lCenter.x(); 
            points[ c*2 + 1 ] = lc[arr[c]].y() - lCenter.y();
         }

         float dz = (lc[4].z() -lc[0].z()) * 0.5;
         TGeoShape* solid = new TGeoArb8(dz, &points[0]);
         volume = new TGeoVolume("ecal oblique prism", solid, GetMedium(kCaloTower));
         caloShapeMap[cell->param()] = volume;
      }
      else {
         volume = volIt->second;
      }      

      HepGeom::Vector3D<float> gCenter;
      CaloCellGeometry::CornersVec const & gc = cell->getCorners();
      for (int c = 0; c < 8; ++c) {
         gCenter += HepGeom::Vector3D<float>(gc[c].x(), gc[c].y(), gc[c].z());
      }
      gCenter *= 0.125;

      TGeoTranslation gtr(gCenter.x(), gCenter.y(), gCenter.z());
      TGeoRotation rot;
      rot.SetAngles(cell->phiPos()*TMath::RadToDeg(), 0, 0);

      TGeoVolume* holder  = GetDaughter(assembly, "side", kCaloTower, detid.zside());
      holder = GetDaughter(holder, "ieta", kCaloTower, detid.ieta());
      std::stringstream nname;
      nname << detid;
      AddLeafNode(holder, volume, nname.str().c_str(), new TGeoCombiTrans(gtr, rot));
   }
}

//______________________________________________________________________________

TGeoHMatrix* getEcalTrans(CaloCellGeometry::CornersVec const & gc)
{

   TEveVector gCenter;
   for (int i = 0; i < 8; ++i)
      gCenter += TEveVector(gc[i].x(), gc[i].y(), gc[i].z());
   gCenter *= 0.125;

  TEveVector tgCenter; // top center 4 corners
   for (int i = 4; i < 8; ++i)
      tgCenter += TEveVector(gc[i].x(), gc[i].y(), gc[i].z());
   tgCenter *= 0.25;


   TEveVector axis = tgCenter - gCenter;
   axis.Normalize();

   TEveTrans tr;
   TVector3 v1t;
   tr.GetBaseVec(1, v1t);


   TEveVector v1(v1t.x(), v1t.y(), v1t.z());
   double dot13 = axis.Dot(v1);
   TEveVector gd = axis;
   gd*= dot13;
   v1 -= gd;
   v1.Normalize();
   TEveVector v2;
   TMath::Cross(v1.Arr(), axis.Arr(), v2.Arr());
   TMath::Cross(axis.Arr(), v1.Arr(), v2.Arr());
   v2.Normalize();

   tr.SetBaseVec(1, v1.fX, v1.fY, v1.fZ);
   tr.SetBaseVec(2, v2.fX, v2.fY, v2.fZ);
   tr.SetBaseVec(3, axis.fX, axis.fY, axis.fZ);
   tr.Move3PF(gCenter.fX, gCenter.fY, gCenter.fZ);

   TGeoHMatrix* out = new TGeoHMatrix();
   tr.SetGeoHMatrix(*out);
   return out;
}

TGeoShape* makeEcalShape(const TruncatedPyramid* cell)
{
   // printf("BEGIN EE SHAPE --------------------------------\n");
   // std::cout << detid << std::endl;
   const HepGeom::Transform3D idtr;
   TruncatedPyramid::Pt3DVec co(8);
   TruncatedPyramid::Pt3D ref;
   TruncatedPyramid::localCorners( co, cell->param(), ref);
   //for( int c = 0; c < 8; ++c)
   //   printf("lc.push_back(TEveVector(%.4f, %.4f, %.4f));\n", co[c].x(),co[c].y(),co[c].z() );

   double points[16];
   for( int c = 0; c < 8; ++c )
   {
      points[c*2  ] = co[c].x();
      points[c*2+1] = co[c].y();
   }
   TGeoShape* solid = new TGeoArb8(cell->param()[0], points);
   return solid;
}

//______________________________________________________________________________



void
FWTGeoRecoGeometryESProducer::addEcalCaloGeometry( void )
{

   TGeoVolume* tv =  GetTopHolder("ECal", kECal);
   CaloVolMap caloShapeMap;

   {
      TGeoVolume *assembly = GetDaughter(tv, "ECalBarrel", kECal);

      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Ecal, EcalSubdetector::EcalBarrel);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
	EBDetId detid(*it);
	const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
	const TruncatedPyramid* cell = dynamic_cast<const TruncatedPyramid*>(cellb);
         if (!cell) { printf("ecalBarrel cell not a TruncatedPyramid !!\n"); return; }

         TGeoVolume* volume = nullptr;
         CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
         if  ( volIt == caloShapeMap.end()) 
         {           
            volume = new TGeoVolume( "EE TruncatedPyramid" , makeEcalShape(cell), GetMedium(kECal));
            caloShapeMap[cell->param()] = volume;
         }
         else {
            volume = volIt->second;
         }
         TGeoHMatrix* mtx= getEcalTrans(cell->getCorners());
         TGeoVolume* holder = GetDaughter(assembly, "side", kECal, detid.zside());
         holder = GetDaughter(holder, "ieta", kECal, detid.ieta());
         std::stringstream nname;
         nname << detid;
         AddLeafNode(holder, volume, nname.str().c_str(), mtx);
      }
   }
   

   {
      TGeoVolume *assembly = GetDaughter(tv, "ECalEndcap", kECal);

      std::vector<DetId> vid = m_caloGeom->getValidDetIds(DetId::Ecal, EcalSubdetector::EcalEndcap);
      for( std::vector<DetId>::const_iterator it = vid.begin(), end = vid.end(); it != end; ++it)
      {
         EEDetId detid(*it);
	 const CaloCellGeometry* cellb = (m_caloGeom->getGeometry(*it)).get();
         const TruncatedPyramid* cell = dynamic_cast<const TruncatedPyramid*>(cellb);
         if (!cell) { printf("ecalEndcap cell not a TruncatedPyramid !!\n"); continue;}

         TGeoVolume* volume = nullptr;
         CaloVolMap::iterator volIt =  caloShapeMap.find(cell->param());
         if  ( volIt == caloShapeMap.end()) 
         {
            
            volume = new TGeoVolume( "EE TruncatedPyramid" , makeEcalShape(cell), GetMedium(kECal));
            caloShapeMap[cell->param()] = volume;
         }
         else {
            volume = volIt->second;
         }
         TGeoHMatrix* mtx= getEcalTrans(cell->getCorners());
         TGeoVolume* holder = GetDaughter(assembly, "side", kECal, detid.zside());
         holder = GetDaughter(holder, "ix", kECal, detid.ix());
         std::stringstream nname;
         nname << detid;
         AddLeafNode(holder, volume, nname.str().c_str(), mtx);
      }
   }
}

