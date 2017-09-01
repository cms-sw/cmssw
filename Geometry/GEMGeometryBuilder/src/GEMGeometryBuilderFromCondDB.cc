/** Implementation of the GEM Geometry Builder from DDD stored in CondDB
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

GEMGeometry* GEMGeometryBuilderFromCondDB::build(const RecoIdealGeometry& rgeo)
{
  const std::vector<DetId>& detids(rgeo.detIds());
  GEMGeometry* geometry = new GEMGeometry();
  
  std::string name;
  std::vector<double>::const_iterator tranStart;
  std::vector<double>::const_iterator shapeStart;
  std::vector<double>::const_iterator rotStart;
  std::vector<std::string>::const_iterator strStart;
  
  for (unsigned int id = 0; id < detids.size(); ++id)
  {  
    GEMDetId gemid(detids[id]);
    //    GEMDetId chid(gemid.region(),gemid.ring(),gemid.station(),
    //		  gemid.sector(),gemid.layer(),gemid.subsector(),0);
    
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
    
    Bounds* bounds = nullptr;
    float be = *(shapeStart+0)/cm;
    float te = *(shapeStart+1)/cm;
    float ap = *(shapeStart+2)/cm;
    float ti = *(shapeStart+3)/cm;
    float nstrip = *(shapeStart+4);
    float npad = *(shapeStart+5);
    //  TrapezoidalPlaneBounds* 
    bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

    std::vector<float> pars;
    pars.emplace_back(be); //b/2;
    pars.emplace_back(te); //B/2;
    pars.emplace_back(ap); //h/2;
    pars.emplace_back(nstrip);
    pars.emplace_back(npad);
    
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
    /*
    if (chids.find(chid)!=chids.end()){
      gepls = chids[chid];
    }
    */
    gepls.emplace_back(gep);
    //chids[chid]=gepls;
    
  }
  /*
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
  */
  return geometry;
}

    

