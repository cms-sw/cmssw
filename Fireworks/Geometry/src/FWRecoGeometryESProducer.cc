#include <fstream>
#include <streambuf>

#include "Fireworks/Geometry/interface/FWRecoGeometryESProducer.h"
#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
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

#include "TNamed.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

void FWRecoGeometryESProducer::ADD_PIXEL_TOPOLOGY( unsigned int rawid,
                                                   const GeomDet* detUnit,
                                                   FWRecoGeometry& fwRecoGeometry ) {
   const PixelGeomDetUnit* det = dynamic_cast<const PixelGeomDetUnit*>( detUnit ); 
   if( det )							
   {      							
      const PixelTopology* topo = &det->specificTopology(); 
    
      std::pair<float,float> pitch = topo->pitch();
      fwRecoGeometry.idToName[rawid].topology[0] = pitch.first;
      fwRecoGeometry.idToName[rawid].topology[1] = pitch.second;

      fwRecoGeometry.idToName[rawid].topology[2] = topo->localX(0.f); // offsetX
      fwRecoGeometry.idToName[rawid].topology[3] = topo->localY(0.f); // offsetY

      // big pixels layout
      fwRecoGeometry.idToName[rawid].topology[4] = topo->isItBigPixelInX(80) ? 0 : 1;
   }
}

# define ADD_SISTRIP_TOPOLOGY( rawid, detUnit )			\
  const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>( detUnit ); \
  if( det )                                                     \
  {                                                             \
    const StripTopology* topo = dynamic_cast<const StripTopology*>( &det->specificTopology() ); \
    fwRecoGeometry.idToName[rawid].topology[0] = 0;            		    \
    fwRecoGeometry.idToName[rawid].topology[1] = topo->nstrips();            \
    fwRecoGeometry.idToName[rawid].topology[2] = topo->stripLength();        \
    if( const RadialStripTopology* rtop = dynamic_cast<const RadialStripTopology*>( &(det->specificType().specificTopology()) ) ) \
      {									\
      fwRecoGeometry.idToName[rawid].topology[0] = 1;			\
      fwRecoGeometry.idToName[rawid].topology[3] = rtop->yAxisOrientation(); \
      fwRecoGeometry.idToName[rawid].topology[4] = rtop->originToIntersection(); \
      fwRecoGeometry.idToName[rawid].topology[5] = rtop->phiOfOneEdge(); \
      fwRecoGeometry.idToName[rawid].topology[6] = rtop->angularWidth(); \
    }                                                                   \
    else if( dynamic_cast<const RectangularStripTopology*>( &(det->specificType().specificTopology()) ) )     \
    {                                                                   \
      fwRecoGeometry.idToName[rawid].topology[0] = 2;			\
      fwRecoGeometry.idToName[rawid].topology[3] = topo->pitch();	\
    }									\
    else if( dynamic_cast<const TrapezoidalStripTopology*>( &(det->specificType().specificTopology()) ) )     \
    {                                                                   \
      fwRecoGeometry.idToName[rawid].topology[0] = 3;			\
      fwRecoGeometry.idToName[rawid].topology[3] = topo->pitch();	\
    }									\
  }                                                                     \

namespace {
  const std::array<std::string,3> hgcal_geom_names =  { { "HGCalEESensitive",
							  "HGCalHESiliconSensitive",
							  "HGCalHEScintillatorSensitive" } };
}
									  
FWRecoGeometryESProducer::FWRecoGeometryESProducer( const edm::ParameterSet& pset )
  : m_current( -1 )
{
  m_tracker = pset.getUntrackedParameter<bool>( "Tracker", true );
  m_muon = pset.getUntrackedParameter<bool>( "Muon", true );
  m_calo = pset.getUntrackedParameter<bool>( "Calo", true );
  m_timing = pset.getUntrackedParameter<bool>( "Timing", false );
  setWhatProduced( this );
}

FWRecoGeometryESProducer::~FWRecoGeometryESProducer( void )
{}

std::unique_ptr<FWRecoGeometry> 
FWRecoGeometryESProducer::produce( const FWRecoGeometryRecord& record )
{
  using namespace edm;

  auto fwRecoGeometry = std::make_unique<FWRecoGeometry>();

  if( m_tracker || m_muon ) {
    record.getRecord<GlobalTrackingGeometryRecord>().get( m_geomRecord );  
    DetId detId( DetId::Tracker, 0 );
    m_trackerGeom = (const TrackerGeometry*) m_geomRecord->slaveGeometry( detId );
  }
    
  if( m_tracker )
  {
    addPixelBarrelGeometry(*fwRecoGeometry);
    addPixelForwardGeometry(*fwRecoGeometry);
    addTIBGeometry(*fwRecoGeometry);
    addTIDGeometry(*fwRecoGeometry);
    addTOBGeometry(*fwRecoGeometry);
    addTECGeometry(*fwRecoGeometry);
    writeTrackerParametersXML(*fwRecoGeometry);
  }
  if( m_muon )
  {
    addDTGeometry(*fwRecoGeometry);
    addCSCGeometry(*fwRecoGeometry);
    addRPCGeometry(*fwRecoGeometry);
    addGEMGeometry(*fwRecoGeometry);
    addME0Geometry(*fwRecoGeometry);
  }
  if( m_calo )
  {
    edm::ESHandle<CaloGeometry>                caloGeomH;
    record.getRecord<CaloGeometryRecord>().get(caloGeomH);
    m_caloGeom = caloGeomH.product();
    addCaloGeometry(*fwRecoGeometry);
  }

  if( m_timing ) {
    record.getRecord<CaloGeometryRecord>().getRecord<IdealGeometryRecord>().get( "FastTimeBarrel", m_ftlBarrelGeom );
    record.getRecord<CaloGeometryRecord>().getRecord<IdealGeometryRecord>().get( "SFBX", m_ftlEndcapGeom );
    addFTLGeometry(*fwRecoGeometry);
  }
  
  fwRecoGeometry->idToName.resize( m_current + 1 );
  std::vector<FWRecoGeom::Info>( fwRecoGeometry->idToName ).swap( fwRecoGeometry->idToName );
  std::sort( fwRecoGeometry->idToName.begin(), fwRecoGeometry->idToName.end());

  return fwRecoGeometry;
}

void
FWRecoGeometryESProducer::addCSCGeometry( FWRecoGeometry& fwRecoGeometry )
{
  DetId detId( DetId::Muon, 2 ); 
  const CSCGeometry* cscGeometry = (const CSCGeometry*) m_geomRecord->slaveGeometry( detId );
  for( auto it = cscGeometry->chambers().begin(),
	   end = cscGeometry->chambers().end(); 
       it != end; ++it )
  {
    const CSCChamber *chamber = *it;
    
    if( chamber )
    {
      unsigned int rawid = chamber->geographicalId();
      unsigned int current =  insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, chamber, fwRecoGeometry );
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
	  unsigned int current = insert_id( rawid, fwRecoGeometry );
	  fillShapeAndPlacement( current, layer, fwRecoGeometry );

	  const CSCStripTopology* stripTopology = layer->geometry()->topology();
	  fwRecoGeometry.idToName[current].topology[0] = stripTopology->yAxisOrientation();
	  fwRecoGeometry.idToName[current].topology[1] = stripTopology->centreToIntersection();
	  fwRecoGeometry.idToName[current].topology[2] = stripTopology->yCentreOfStripPlane();
	  fwRecoGeometry.idToName[current].topology[3] = stripTopology->phiOfOneEdge();
	  fwRecoGeometry.idToName[current].topology[4] = stripTopology->stripOffset();
	  fwRecoGeometry.idToName[current].topology[5] = stripTopology->angularWidth();

	  const CSCWireTopology* wireTopology = layer->geometry()->wireTopology();
	  fwRecoGeometry.idToName[current].topology[6] = wireTopology->wireSpacing();
	  fwRecoGeometry.idToName[current].topology[7] = wireTopology->wireAngle();
	}
      }
    }
  }
}

void
FWRecoGeometryESProducer::addDTGeometry( FWRecoGeometry& fwRecoGeometry )
{
  DetId detId( DetId::Muon, 1 );
  const DTGeometry* dtGeometry = (const DTGeometry*) m_geomRecord->slaveGeometry( detId );

  //
  // DT chambers geometry
  //
  for( auto it = dtGeometry->chambers().begin(),
	   end = dtGeometry->chambers().end(); 
       it != end; ++it )
  {
    const DTChamber *chamber = *it;
    
    if( chamber )
    {
      unsigned int rawid = chamber->geographicalId().rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, chamber, fwRecoGeometry );
    }
  }

  // Fill in DT layer parameters
  for( auto it = dtGeometry->layers().begin(),
	   end = dtGeometry->layers().end(); 
       it != end; ++it )
  {
    const DTLayer* layer = *it;
     
    if( layer )
    {
      unsigned int rawid = layer->id().rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, layer, fwRecoGeometry );

      const DTTopology& topo = layer->specificTopology();
      const BoundPlane& surf = layer->surface();
      // Topology W/H/L:
      fwRecoGeometry.idToName[current].topology[0] = topo.cellWidth();
      fwRecoGeometry.idToName[current].topology[1] = topo.cellHeight();
      fwRecoGeometry.idToName[current].topology[2] = topo.cellLenght();
      fwRecoGeometry.idToName[current].topology[3] = topo.firstChannel();
      fwRecoGeometry.idToName[current].topology[4] = topo.lastChannel();
      fwRecoGeometry.idToName[current].topology[5] = topo.channels();

      // Bounds W/H/L:
      fwRecoGeometry.idToName[current].topology[6] = surf.bounds().width();
      fwRecoGeometry.idToName[current].topology[7] = surf.bounds().thickness();
      fwRecoGeometry.idToName[current].topology[8] = surf.bounds().length();
    }
  }  
}

void
FWRecoGeometryESProducer::addRPCGeometry( FWRecoGeometry& fwRecoGeometry )
{
  //
  // RPC rolls geometry
  //
  DetId detId( DetId::Muon, 3 );
  const RPCGeometry* rpcGeom = (const RPCGeometry*) m_geomRecord->slaveGeometry( detId );
  for( auto it = rpcGeom->rolls().begin(),
	   end = rpcGeom->rolls().end(); 
       it != end; ++it )
  {
    const RPCRoll* roll = (*it);
    if( roll )
    {
      unsigned int rawid = roll->geographicalId().rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, roll, fwRecoGeometry );

      const StripTopology& topo = roll->specificTopology();
      fwRecoGeometry.idToName[current].topology[0] = topo.nstrips();
      fwRecoGeometry.idToName[current].topology[1] = topo.stripLength();
      fwRecoGeometry.idToName[current].topology[2] = topo.pitch();
    }
  }


  try {
     RPCDetId id(1, 1, 4, 1, 1, 1, 1 );
     m_geomRecord->slaveGeometry( detId );
     fwRecoGeometry.extraDet.Add(new TNamed("RE4", "RPC endcap station 4"));
  }
    catch (std::runtime_error &e) {
       std::cerr << e.what() << std::endl; 
    }
}

void
FWRecoGeometryESProducer::addGEMGeometry( FWRecoGeometry& fwRecoGeometry )
{
  //
  // GEM geometry
  //
  DetId detId( DetId::Muon, 4 );

  try 
  {
    const GEMGeometry* gemGeom = (const GEMGeometry*) m_geomRecord->slaveGeometry( detId );
    
    // add in superChambers - gem Segments are based on superChambers
    for(auto sc : gemGeom->superChambers())
    { 
      if( sc )
      {
	unsigned int rawid = sc->geographicalId().rawId();
	unsigned int current = insert_id( rawid, fwRecoGeometry );
	fillShapeAndPlacement( current, sc, fwRecoGeometry );
      }
    }
    // add in chambers
    for(auto ch : gemGeom->chambers())
    { 
      if( ch )
      {
	unsigned int rawid = ch->geographicalId().rawId();
	unsigned int current = insert_id( rawid, fwRecoGeometry );
	fillShapeAndPlacement( current, ch, fwRecoGeometry );
      }
    }    
    // add in etaPartitions - gem rechits are based on etaPartitions
    for(auto roll : gemGeom->etaPartitions())
    { 
      if( roll )
      {
	unsigned int rawid = roll->geographicalId().rawId();
	unsigned int current = insert_id( rawid, fwRecoGeometry );
	fillShapeAndPlacement( current, roll, fwRecoGeometry );

	const StripTopology& topo = roll->specificTopology();
	fwRecoGeometry.idToName[current].topology[0] = topo.nstrips();
	fwRecoGeometry.idToName[current].topology[1] = topo.stripLength();
	fwRecoGeometry.idToName[current].topology[2] = topo.pitch();

	float height = topo.stripLength()/2;
	LocalPoint  lTop( 0., height, 0.);
	LocalPoint  lBottom( 0., -height, 0.);
	fwRecoGeometry.idToName[current].topology[3] = roll->localPitch(lTop);
	fwRecoGeometry.idToName[current].topology[4] = roll->localPitch(lBottom);
	fwRecoGeometry.idToName[current].topology[5] = roll->npads();
      }
    }

    fwRecoGeometry.extraDet.Add(new TNamed("GEM", "GEM muon detector"));
    try {
      GEMDetId id(1, 1, 2, 1, 1, 1 );
      m_geomRecord->slaveGeometry( detId );
      fwRecoGeometry.extraDet.Add(new TNamed("GE2", "GEM endcap station 2"));
    }
    catch (std::runtime_error &e) {
       std::cerr << e.what() << std::endl; 
    }

  }
  catch( cms::Exception &exception )
  {
    edm::LogError("FWRecoGeometry") << " GEM geometry not found " << exception.what() << std::endl;
  }
}

void
FWRecoGeometryESProducer::addME0Geometry( FWRecoGeometry& fwRecoGeometry )
{
  //                                                                                                                               
  // ME0 geometry                                                                                                                  
  //   

  DetId detId( DetId::Muon, 5 );
  try 
  {
    const ME0Geometry* me0Geom = (const ME0Geometry*) m_geomRecord->slaveGeometry( detId );
    for(auto roll : me0Geom->etaPartitions())
    { 
      if( roll )
      {
	unsigned int rawid = roll->geographicalId().rawId();
	unsigned int current = insert_id( rawid, fwRecoGeometry );
	fillShapeAndPlacement( current, roll, fwRecoGeometry );
	  
	const StripTopology& topo = roll->specificTopology();
	fwRecoGeometry.idToName[current].topology[0] = topo.nstrips();
	fwRecoGeometry.idToName[current].topology[1] = topo.stripLength();
	fwRecoGeometry.idToName[current].topology[2] = topo.pitch();
	
	float height = topo.stripLength()/2;
	LocalPoint  lTop( 0., height, 0.);
	LocalPoint  lBottom( 0., -height, 0.);
	fwRecoGeometry.idToName[current].topology[3] = roll->localPitch(lTop);
	fwRecoGeometry.idToName[current].topology[4] = roll->localPitch(lBottom);
	fwRecoGeometry.idToName[current].topology[5] = roll->npads();
      }
    }
    fwRecoGeometry.extraDet.Add(new TNamed("ME0", "ME0 muon detector"));
  }
  catch( cms::Exception &exception )
  {
    edm::LogError("FWRecoGeometry") << " ME0 geometry not found " << exception.what() << std::endl;
  }
}  

void
FWRecoGeometryESProducer::addPixelBarrelGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXB().begin(),
						    end = m_trackerGeom->detsPXB().end();
       it != end; ++it)
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );

      ADD_PIXEL_TOPOLOGY( current, m_trackerGeom->idToDetUnit( detid ), fwRecoGeometry);
    }
  }
}

void
FWRecoGeometryESProducer::addPixelForwardGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXF().begin(),
						    end = m_trackerGeom->detsPXF().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );
    
      ADD_PIXEL_TOPOLOGY( current, m_trackerGeom->idToDetUnit( detid ), fwRecoGeometry);
    }
  }
}

void
FWRecoGeometryESProducer::addTIBGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTIB().begin(),
						    end = m_trackerGeom->detsTIB().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
}

void
FWRecoGeometryESProducer::addTOBGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTOB().begin(),
						    end = m_trackerGeom->detsTOB().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
}

void
FWRecoGeometryESProducer::addTIDGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTID().begin(),
						    end = m_trackerGeom->detsTID().end();
       it != end; ++it)
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
}

void
FWRecoGeometryESProducer::addTECGeometry( FWRecoGeometry& fwRecoGeometry )
{
  for( TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTEC().begin(),
						    end = m_trackerGeom->detsTEC().end();
       it != end; ++it )
  {
    const GeomDet *det = *it;

    if( det )
    {     
      DetId detid = det->geographicalId();
      unsigned int rawid = detid.rawId();
      unsigned int current = insert_id( rawid, fwRecoGeometry );
      fillShapeAndPlacement( current, det, fwRecoGeometry );

      ADD_SISTRIP_TOPOLOGY( current, m_trackerGeom->idToDet( detid ));
    }
  }
}

void
FWRecoGeometryESProducer::addCaloGeometry( FWRecoGeometry& fwRecoGeometry )
{
  std::vector<DetId> vid = m_caloGeom->getValidDetIds(); // Calo
  for( std::vector<DetId>::const_iterator it = vid.begin(),
	 end = vid.end();
       it != end; ++it ) {
    unsigned int id = insert_id( it->rawId(), fwRecoGeometry);
    if( !((DetId::Forward == it->det()) || (DetId::HGCalEE == it->det()) ||
	  (DetId::HGCalHSi == it->det()) || (DetId::HGCalHSc == it->det())) ) {
      const CaloCellGeometry::CornersVec& cor = m_caloGeom->getGeometry( *it )->getCorners();
      fillPoints( id, cor.begin(), cor.end(), fwRecoGeometry);
    } else {
      DetId::Detector det = it->det();
      int          subdet = (((DetId::HGCalEE == det) || 
			     (DetId::HGCalHSi == det) ||
			     (DetId::HGCalHSc == det)) ? ForwardEmpty :
			     it->subdetId());
      const HGCalGeometry* geom = dynamic_cast<const HGCalGeometry*>(m_caloGeom->getSubdetectorGeometry(det,subdet));
      const auto cor = geom->getNewCorners( *it );

      // roll = yaw = pitch = 0
      fwRecoGeometry.idToName[id].matrix[0] = 1.0;
      fwRecoGeometry.idToName[id].matrix[4] = 1.0;
      fwRecoGeometry.idToName[id].matrix[8] = 1.0;

      // corners of the front face
      for(uint i = 0; i < (cor.size()-1); ++i){
        fwRecoGeometry.idToName[id].points[i*3+0] = cor[i].x();
        fwRecoGeometry.idToName[id].points[i*3+1] = cor[i].y();
        fwRecoGeometry.idToName[id].points[i*3+2] = cor[i].z();
      }

      // thickness
      fwRecoGeometry.idToName[id].shape[3] = cor[cor.size()-1].z();

      // total points
      fwRecoGeometry.idToName[id].topology[0] = cor.size() - 1;
    }
  }
}

void
FWRecoGeometryESProducer::addFTLGeometry( FWRecoGeometry& fwRecoGeometry )
{
  // do the barrel
  std::vector<DetId> vid = std::move(m_ftlBarrelGeom->getValidDetIds()); 
  for( std::vector<DetId>::const_iterator it = vid.begin(),
	 end = vid.end();
       it != end; ++it ) {
    unsigned int id = insert_id( it->rawId(), fwRecoGeometry);
    const auto& cor =  m_ftlBarrelGeom->getCorners( *it );      
    fillPoints( id, cor.begin(), cor.end(), fwRecoGeometry);    
  }
  // do the endcap
  vid = std::move(m_ftlEndcapGeom->getValidDetIds()); 
  for( std::vector<DetId>::const_iterator it = vid.begin(),
	 end = vid.end();
       it != end; ++it ) {
    unsigned int id = insert_id( it->rawId(), fwRecoGeometry);
    const auto& cor =  m_ftlEndcapGeom->getCorners( *it );      
    fillPoints( id, cor.begin(), cor.end(), fwRecoGeometry);    
  }
}

unsigned int
FWRecoGeometryESProducer::insert_id( unsigned int rawid,
                                     FWRecoGeometry& fwRecoGeometry )
{
  ++m_current;
  fwRecoGeometry.idToName.push_back(FWRecoGeom::Info());
  fwRecoGeometry.idToName.back().id = rawid;
  
  return m_current;
}

void
FWRecoGeometryESProducer::fillPoints( unsigned int id,
                                      std::vector<GlobalPoint>::const_iterator begin,
                                      std::vector<GlobalPoint>::const_iterator end,
                                      FWRecoGeometry& fwRecoGeometry )
{
  unsigned int index( 0 );
  for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
  {
    assert( index < FWTGeoRecoGeometry::maxPoints_-1 );
    fwRecoGeometry.idToName[id].points[index] = i->x();
    fwRecoGeometry.idToName[id].points[++index] = i->y();
    fwRecoGeometry.idToName[id].points[++index] = i->z();
    ++index;
  }
}


/** Shape of GeomDet */
void
FWRecoGeometryESProducer::fillShapeAndPlacement( unsigned int id, const GeomDet *det,
                                                 FWRecoGeometry& fwRecoGeometry )
{
  // Trapezoidal
  const Bounds *b = &((det->surface ()).bounds ());
  if( const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *> (b))
  {
      std::array< const float, 4 > const & par = b2->parameters ();
    
    // These parameters are half-lengths, as in CMSIM/GEANT3
    fwRecoGeometry.idToName[id].shape[0] = 1;
    fwRecoGeometry.idToName[id].shape[1] = par [0]; // hBottomEdge
    fwRecoGeometry.idToName[id].shape[2] = par [1]; // hTopEdge
    fwRecoGeometry.idToName[id].shape[3] = par [2]; // thickness
    fwRecoGeometry.idToName[id].shape[4] = par [3]; // apothem
  }
  if( const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *> (b))
  {
    // Rectangular
    fwRecoGeometry.idToName[id].shape[0] = 2;
    fwRecoGeometry.idToName[id].shape[1] = b2->width() * 0.5; // half width
    fwRecoGeometry.idToName[id].shape[2] = b2->length() * 0.5; // half length
    fwRecoGeometry.idToName[id].shape[3] = b2->thickness() * 0.5; // half thickness
  }

  // Position of the DetUnit's center
  GlobalPoint pos = det->surface().position();
  fwRecoGeometry.idToName[id].translation[0] = pos.x();
  fwRecoGeometry.idToName[id].translation[1] = pos.y();
  fwRecoGeometry.idToName[id].translation[2] = pos.z();

  // Add the coeff of the rotation matrix
  // with a projection on the basis vectors
  TkRotation<float> detRot = det->surface().rotation();
  fwRecoGeometry.idToName[id].matrix[0] = detRot.xx();
  fwRecoGeometry.idToName[id].matrix[1] = detRot.yx();
  fwRecoGeometry.idToName[id].matrix[2] = detRot.zx();
  fwRecoGeometry.idToName[id].matrix[3] = detRot.xy();
  fwRecoGeometry.idToName[id].matrix[4] = detRot.yy();
  fwRecoGeometry.idToName[id].matrix[5] = detRot.zy();
  fwRecoGeometry.idToName[id].matrix[6] = detRot.xz();
  fwRecoGeometry.idToName[id].matrix[7] = detRot.yz();
  fwRecoGeometry.idToName[id].matrix[8] = detRot.zz();
}

void FWRecoGeometryESProducer::writeTrackerParametersXML(FWRecoGeometry& fwRecoGeometry)
{
  std::string path = "Geometry/TrackerCommonData/data/";
  if ( m_trackerGeom->isThere(GeomDetEnumerators::P1PXB) ||
       m_trackerGeom->isThere(GeomDetEnumerators::P1PXEC) ) {
    path += "PhaseI/";
  } else if ( m_trackerGeom->isThere(GeomDetEnumerators::P2PXB)  ||
              m_trackerGeom->isThere(GeomDetEnumerators::P2PXEC) ||
              m_trackerGeom->isThere(GeomDetEnumerators::P2OTB)  ||
              m_trackerGeom->isThere(GeomDetEnumerators::P2OTEC) ) {
    path += "PhaseII/";
  }
  path += "trackerParameters.xml";
  std::string fullPath = edm::FileInPath(path).fullPath();
  std::ifstream t(fullPath);
  std::stringstream buffer;
  buffer << t.rdbuf();
  fwRecoGeometry.trackerTopologyXML = buffer.str();
}
