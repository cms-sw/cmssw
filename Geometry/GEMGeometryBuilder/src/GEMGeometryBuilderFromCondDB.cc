/** Implementation of the GEM Geometry Builder from GEM record stored in CondDB
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <algorithm>

GEMGeometryBuilderFromCondDB::GEMGeometryBuilderFromCondDB() 
{ }

GEMGeometryBuilderFromCondDB::~GEMGeometryBuilderFromCondDB() 
{ }

void
GEMGeometryBuilderFromCondDB::build(const std::shared_ptr<GEMGeometry>& theGeometry,
				    const RecoIdealGeometry& rgeo )
{
  const std::vector<DetId>& detids( rgeo.detIds());
  std::vector<GEMSuperChamber*> superChambers;

  std::string name;
  std::vector<double>::const_iterator tranStart;
  std::vector<double>::const_iterator shapeStart;
  std::vector<double>::const_iterator rotStart;
  std::vector<std::string>::const_iterator strStart;

  for( unsigned int id = 0; id < detids.size(); ++id )
  {  
    GEMDetId gemid( detids[id] );
    GEMDetId chid( gemid.region(), gemid.ring(), gemid.station(),
		   gemid.layer(), gemid.chamber(), 0 );

    tranStart = rgeo.tranStart( id );
    shapeStart = rgeo.shapeStart( id );
    rotStart = rgeo.rotStart( id );
    strStart = rgeo.strStart( id );
    name = *( strStart );

    Surface::PositionType pos(*(tranStart)/cm, *(tranStart+1)/cm, *(tranStart+2)/cm );
    // CLHEP way
    Surface::RotationType rot(*(rotStart+0), *(rotStart+1), *(rotStart+2),
                              *(rotStart+3), *(rotStart+4), *(rotStart+5),
                              *(rotStart+6), *(rotStart+7), *(rotStart+8));
    
    float be = *(shapeStart+0)/cm;
    float te = *(shapeStart+1)/cm;
    float ap = *(shapeStart+2)/cm;
    float ti = *(shapeStart+3)/cm;
    float nstrip = *(shapeStart+4);
    float npad = *(shapeStart+5);
    //  TrapezoidalPlaneBounds* 
    Bounds* bounds = new TrapezoidalPlaneBounds( be, te, ap, ti );

    std::vector<float> pars;
    pars.emplace_back(be); //b/2;
    pars.emplace_back(te); //B/2;
    pars.emplace_back(ap); //h/2;
    pars.emplace_back(nstrip);
    pars.emplace_back(npad);
    
    GEMEtaPartitionSpecs* epSpecs = new GEMEtaPartitionSpecs( GeomDetEnumerators::GEM, name, pars );
      
    //Change of axes for the forward
    Basic3DVector<float> newX( 1., 0., 0. );
    Basic3DVector<float> newY( 0., 0., 1. );
    //      if (tran[2] > 0. )
    newY *= -1;
    Basic3DVector<float> newZ( 0., 1., 0. );
    rot.rotateAxes( newX, newY, newZ );	
  
    BoundPlane* bp = new BoundPlane( pos, rot, bounds );
    ReferenceCountingPointer<BoundPlane> surf( bp );
    GEMEtaPartition* gep = new GEMEtaPartition( gemid, surf, epSpecs );
    LogDebug("GEMGeometryBuilder") << "GEM Eta Partition created with id = " << gemid
				   << " and added to the GEMGeometry" << std::endl;
    theGeometry->add(gep);
    
    std::list<GEMEtaPartition *> gepList;
    if( m_chids.find( chid ) != m_chids.end()) {
      gepList = m_chids[chid];
    }
    gepList.emplace_back(gep);
    m_chids[chid] = gepList;
  }
  
  // Create the GEMChambers and store them on the Geometry 

  for( const auto& ich : m_chids ) {
    GEMDetId chid = ich.first;
    std::list<GEMEtaPartition * > gepList = ich.second;

    // compute the overall boundplane. At the moment we use just the last
    // surface
    BoundPlane* bp = nullptr;
    for( const auto& gep : gepList ) {
      const BoundPlane& bps = ( *gep ).surface();
      bp = const_cast<BoundPlane *>( &bps );
    }

    ReferenceCountingPointer<BoundPlane> surf( bp );
    // Create the superchamber
    if( chid.layer() == 1 ) {
      GEMDetId schid( chid.region(), chid.ring(), chid.station(), 0, chid.chamber(), 0 );
      GEMSuperChamber* sch = new GEMSuperChamber( schid, surf );
      LogDebug("GEMGeometryBuilder") << "GEM SuperChamber created with id = " << schid
				     << " and added to the GEMGeometry" << std::endl;
      superChambers.emplace_back( sch );
    }
    
    // Create the chamber 
    GEMChamber* ch = new GEMChamber( chid, surf ); 
    LogDebug("GEMGeometryBuilder") << "GEM Chamber created with id = " << chid
				   << " = " << chid.rawId() << " and added to the GEMGeometry" << std::endl;
    LogDebug("GEMGeometryBuilder") << "GEM Chamber has following eta partitions associated: " << std::endl;

    // Add the etaps to rhe chamber
    for( const auto& gep : gepList ) {
      ch->add(gep);
      LogDebug("GEMGeometryBuilder") << "   --> GEM Eta Partition " << GEMDetId(( *gep ).id()) << std::endl;
    }
    // Add the chamber to the geometry
    theGeometry->add( ch );
  }
  
  // The super chamber is composed of 2 chambers.
  // It's detId is layer 0, chambers are layer 1 and 2

  // construct the regions, stations and rings. 
  for( int re = -1; re <= 1; re = re+2 ) {
    GEMRegion* region = new GEMRegion( re );
    for( int st = 1; st <= GEMDetId::maxStationId; ++st ) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign( re==-1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      station->setName(name);
      for( int ri = 1; ri <= 1; ++ri ) {
	GEMRing* ring = new GEMRing( re, st, ri );
	for( unsigned sch = 0; sch < superChambers.size(); ++sch ) {
	  GEMSuperChamber* superChamber = superChambers[sch];
	  const GEMDetId detId( superChamber->id());
	  if (detId.region() != re || detId.station() != st || detId.ring() != ri) continue;
	  
	  superChamber->add( theGeometry->chamber( GEMDetId( detId.region(), detId.ring(), detId.station(), 1, detId.chamber(), 0 )));
	  superChamber->add( theGeometry->chamber( GEMDetId( detId.region(), detId.ring(), detId.station(), 2, detId.chamber(), 0 )));
	  ring->add( superChamber );
	  theGeometry->add( superChamber );
	  LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: " 
						<< "re " << re << " st " << st << " ri " << ri << std::endl;
	}
	LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " <<  ri << " to station " << "re " << re << " st " << st << std::endl;
	station->add( ring );
	theGeometry->add( ring );
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add( station );
      theGeometry->add( station );
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;
    theGeometry->add( region );
  }
}
