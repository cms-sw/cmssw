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

#include <iostream>
#include <algorithm>

GEMGeometryBuilderFromCondDB::GEMGeometryBuilderFromCondDB() 
{ }

GEMGeometryBuilderFromCondDB::~GEMGeometryBuilderFromCondDB() 
{ }

GEMGeometry*
GEMGeometryBuilderFromCondDB::build( const RecoIdealGeometry& rgeo )
{
  const std::vector<DetId>& detids(rgeo.detIds());
  GEMGeometry* geometry = new GEMGeometry();
  
  std::string name;
  std::vector<double>::const_iterator tranStart;
  std::vector<double>::const_iterator shapeStart;
  std::vector<double>::const_iterator rotStart;
  std::vector<std::string>::const_iterator strStart;

  GEMChamber* chamber(0);
  GEMSuperChamber* sch(0);
  
  for (unsigned int id = 0; id < detids.size(); ++id)
  {  
    GEMDetId gemid(detids[id]);
    GEMDetId chid(gemid.region(),gemid.ring(),gemid.station(),
		  gemid.layer(),gemid.chamber(),0);
    
    tranStart = rgeo.tranStart(id);
    shapeStart = rgeo.shapeStart(id);
    rotStart = rgeo.rotStart(id);
    strStart = rgeo.strStart(id);
    name = *(strStart);

    Surface::PositionType pos(*(tranStart)/cm,*(tranStart+1)/cm, *(tranStart+2)/cm);
    // CLHEP way
    Surface::RotationType rot(*(rotStart+0), *(rotStart+1), *(rotStart+2),
                              *(rotStart+3), *(rotStart+4), *(rotStart+5),
                              *(rotStart+6), *(rotStart+7), *(rotStart+8));
    
    Bounds* bounds = 0;
    float be = *(shapeStart+0)/cm;
    float te = *(shapeStart+1)/cm;
    float ap = *(shapeStart+2)/cm;
    float ti = *(shapeStart+3)/cm;
    float nstrip = *(shapeStart+4);
    float npad = *(shapeStart+5);
    //  TrapezoidalPlaneBounds* 
    bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

    std::vector<float> pars;
    pars.push_back(be); //b/2;
    pars.push_back(te); //B/2;
    pars.push_back(ap); //h/2;
    pars.push_back(nstrip);
    pars.push_back(npad);
    
    GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);
      
    //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);
    //      if (tran[2] > 0. )
    newY *= -1;
    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY, newZ);	
  
    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    GEMEtaPartition* gep=new GEMEtaPartition(gemid, surf, e_p_specs);
    geometry->add(gep);
    
    std::list<GEMEtaPartition *> gepls;
    
    if( chids.find(chid) != chids.end()) {
      gepls = chids[chid];
    }
    
    gepls.push_back(gep);
    chids[chid] = gepls;

  }
  
  // Create the GEMChambers and store them on the Geometry 

  for( std::map<GEMDetId, std::list<GEMEtaPartition *> >::iterator ich=chids.begin();
       ich != chids.end(); ich++){
    GEMDetId chid = ich->first;
    std::list<GEMEtaPartition * > gepls = ich->second;

    // compute the overall boundplane. At the moment we use just the last
    // surface
    BoundPlane* bp=0;
    for(std::list<GEMEtaPartition *>::iterator gepl=gepls.begin();
    gepl!=gepls.end(); gepl++){
    const BoundPlane& bps = (*gepl)->surface();
      bp = const_cast<BoundPlane *>(&bps);
    }

    ReferenceCountingPointer<BoundPlane> surf(bp);
    // Create the chamber 
    GEMChamber* ch = new GEMChamber (chid, surf); 
    // Add the etaps to rhe chamber
    for(std::list<GEMEtaPartition *>::iterator gepl=gepls.begin();
    gepl!=gepls.end(); gepl++){
      ch->add(*gepl);
    }
    // Add the chamber to the geometry
    geometry->add(ch);

  }
  
  // The super chamber is composed of 2 chambers.
  // It's detId is layer 0, chambers are layer 1 and 2

  auto& superChambers( geometry->superChambers());
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
	  GEMSuperChamber* superChamber = const_cast<GEMSuperChamber*>( superChambers.at(sch));
	  const GEMDetId detId( superChamber->id());
	  if (detId.region() != re || detId.station() != st || detId.ring() != ri) continue;
	  
	  auto ch1 = geometry->chamber(GEMDetId(detId.region(),detId.ring(),detId.station(),1,detId.chamber(),0));
	  auto ch2 = geometry->chamber(GEMDetId(detId.region(),detId.ring(),detId.station(),2,detId.chamber(),0));
	  superChamber->add(const_cast<GEMChamber*>(ch1));
	  superChamber->add(const_cast<GEMChamber*>(ch2));
       
	  ring->add(superChamber);
	  LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: " 
						<< "re " << re << " st " << st << " ri " << ri << std::endl;
	}
	LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " <<  ri << " to station " << "re " << re << " st " << st << std::endl;
	station->add(const_cast<GEMRing*>(ring));
	geometry->add(const_cast<GEMRing*>(ring));
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add(const_cast<GEMStation*>(station));
      geometry->add(const_cast<GEMStation*>(station));
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;
    geometry->add(const_cast<GEMRegion*>(region));
  }

  return geometry;
}

GEMSuperChamber*
GEMGeometryBuilderFromCondDB::buildSuperChamber( const GEMDetId id, const RecoIdealGeometry& rig,
						 size_t idt ) const
{
  RCPPlane surf;
  GEMSuperChamber* chamber = new GEMSuperChamber( id, surf );

  return chamber;
}
  
GEMChamber*
GEMGeometryBuilderFromCondDB::buildChamber( const GEMDetId id,  const RecoIdealGeometry& rig,
					    size_t idt ) const
{
  RCPPlane surf;
  GEMChamber* chamber = new GEMChamber( id, surf );

  return chamber;
}


