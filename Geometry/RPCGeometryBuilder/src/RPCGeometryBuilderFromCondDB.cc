/** Implementation of the RPC Geometry Builder from DDD stored in CondDB
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

RPCGeometryBuilderFromCondDB::RPCGeometryBuilderFromCondDB(bool comp11) :
  theComp11Flag(comp11)
{ }

RPCGeometryBuilderFromCondDB::~RPCGeometryBuilderFromCondDB() 
{ }

RPCGeometry* RPCGeometryBuilderFromCondDB::build(const RecoIdealGeometry& rgeo)
{
  const std::vector<DetId>& detids(rgeo.detIds());
  RPCGeometry* geometry = new RPCGeometry();
  
  std::string name;
  std::vector<double>::const_iterator tranStart;
  std::vector<double>::const_iterator shapeStart;
  std::vector<double>::const_iterator rotStart;
  std::vector<std::string>::const_iterator strStart;
  
  for (unsigned int id=0; id<detids.size(); id++){
    
    RPCDetId rpcid(detids[id]);
    RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),
		  rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);
    
    tranStart=rgeo.tranStart(id);
    shapeStart=rgeo.shapeStart(id);
    rotStart=rgeo.rotStart(id);
    strStart=rgeo.strStart(id);
    name=*(strStart);

    Surface::PositionType pos(*(tranStart)/cm,*(tranStart+1)/cm, *(tranStart+2)/cm);
    // CLHEP way
    Surface::RotationType rot(*(rotStart+0),*(rotStart+1),*(rotStart+2),
			      *(rotStart+3),*(rotStart+4),*(rotStart+5),
			      *(rotStart+6),*(rotStart+7),*(rotStart+8));
      
    std::vector<float> pars;
    RPCRollSpecs* rollspecs= 0;
    Bounds* bounds = 0;
    
    //    if (dpar.size()==4){
    if ( rgeo.shapeEnd(id) - shapeStart == 4 ) {
      float width     = *(shapeStart+0)/cm;
      float length    = *(shapeStart+1)/cm;
      float thickness = *(shapeStart+2)/cm;
      float nstrip    = *(shapeStart+3);
      //RectangularPlaneBounds* 
      bounds = 
	new RectangularPlaneBounds(width,length,thickness);
      pars.push_back(width);
      pars.push_back(length);
      pars.push_back(nstrip); 

      if (!theComp11Flag) {
	//Correction of the orientation to get the REAL geometry.
        //Change of axes for the +z part only.
        //Including the 0 whell
        if ( *(tranStart+2) > -1500.) {   //tran[2] >-1500. ){
          Basic3DVector<float> newX(-1.,0.,0.);
          Basic3DVector<float> newY(0.,-1.,0.);
          Basic3DVector<float> newZ(0.,0.,1.);
          rot.rotateAxes (newX, newY,newZ);
        }
      }

      
      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel,name,pars);
      
      
    }else{
      float be = *(shapeStart+0)/cm;
      float te = *(shapeStart+1)/cm;
      float ap = *(shapeStart+2)/cm;
      float ti = *(shapeStart+3)/cm;
      float nstrip = *(shapeStart+4);
      //  TrapezoidalPlaneBounds* 
      bounds = 
	new TrapezoidalPlaneBounds(be,te,ap,ti);
      pars.push_back(be); //b/2;
      pars.push_back(te); //B/2;
      pars.push_back(ap); //h/2;
      pars.push_back(nstrip); 
      
      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap,name,pars);
      
      //Change of axes for the forward
      Basic3DVector<float> newX(1.,0.,0.);
      Basic3DVector<float> newY(0.,0.,1.);
      //      if (tran[2] > 0. )
      newY *= -1;
      Basic3DVector<float> newZ(0.,1.,0.);
      rot.rotateAxes (newX, newY,newZ);	
    }
    
    BoundPlane* bp = new BoundPlane(pos,rot,bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    RPCRoll* r=new RPCRoll(rpcid,surf,rollspecs);
    geometry->add(r);
    
    std::list<RPCRoll *> rls;
    if (chids.find(chid)!=chids.end()){
      rls = chids[chid];
    }
    rls.push_back(r);
    chids[chid]=rls;
  }
  // Create the RPCChambers and store them on the Geometry 
  for( std::map<RPCDetId, std::list<RPCRoll *> >::iterator ich=chids.begin();
       ich != chids.end(); ich++){
    RPCDetId chid = ich->first;
    std::list<RPCRoll * > rls = ich->second;

    // compute the overall boundplane. At the moment we use just the last
    // surface
    BoundPlane* bp=0;
    for(std::list<RPCRoll *>::iterator rl=rls.begin();
    rl!=rls.end(); rl++){
      const BoundPlane& bps = (*rl)->surface();
      bp = const_cast<BoundPlane *>(&bps);
    }

    ReferenceCountingPointer<BoundPlane> surf(bp);
    // Create the chamber 
    RPCChamber* ch = new RPCChamber (chid, surf); 
    // Add the rolls to rhe chamber
    for(std::list<RPCRoll *>::iterator rl=rls.begin();
    rl!=rls.end(); rl++){
      ch->add(*rl);
    }
    // Add the chamber to the geometry
    geometry->add(ch);

  }
  return geometry;
}

    

