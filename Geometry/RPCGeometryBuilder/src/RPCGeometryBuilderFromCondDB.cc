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
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

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

  for (unsigned int id=0; id<detids.size(); ++id){
    RPCDetId rpcid(detids[id]);
    RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),
                  rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);

    const auto tranStart  = rgeo.tranStart(id);
    const auto shapeStart = rgeo.shapeStart(id);
    const auto rotStart   = rgeo.rotStart(id);
    const std::string& name = *rgeo.strStart(id);

    Surface::PositionType pos(*(tranStart)/cm,*(tranStart+1)/cm, *(tranStart+2)/cm);
    // CLHEP way
    Surface::RotationType rot(*(rotStart+0),*(rotStart+1),*(rotStart+2),
                              *(rotStart+3),*(rotStart+4),*(rotStart+5),
                              *(rotStart+6),*(rotStart+7),*(rotStart+8));

    RPCRollSpecs* rollspecs= nullptr;
    Bounds* bounds = nullptr;

    //    if (dpar.size()==4){
    if ( rgeo.shapeEnd(id) - shapeStart == 4 ) {
      const float width     = *(shapeStart+0)/cm;
      const float length    = *(shapeStart+1)/cm;
      const float thickness = *(shapeStart+2)/cm;
      const float nstrip    = *(shapeStart+3);
      //RectangularPlaneBounds*
      bounds = new RectangularPlaneBounds(width,length,thickness);
      const std::vector<float> pars = {width, length, nstrip};

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

    }
    else {
      const float be = *(shapeStart+0)/cm;
      const float te = *(shapeStart+1)/cm;
      const float ap = *(shapeStart+2)/cm;
      const float ti = *(shapeStart+3)/cm;
      const float nstrip = *(shapeStart+4);
      //  TrapezoidalPlaneBounds*
      bounds = new TrapezoidalPlaneBounds(be,te,ap,ti);
      const std::vector<float> pars = {be /*b/2*/, te /*B/2*/, ap /*h/2*/, nstrip};

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
    RPCRoll* r = new RPCRoll(rpcid,surf,rollspecs);
    geometry->add(r);

    auto rls = chids.find(chid);
    if ( rls == chids.end() ) rls = chids.insert(std::make_pair(chid, std::list<RPCRoll*>())).first;
    rls->second.emplace_back(r);
  }

  // Create the RPCChambers and store them on the Geometry
  for (auto & ich : chids) {
    const RPCDetId& chid = ich.first;
    const auto& rls = ich.second;

    // compute the overall boundplane.
    BoundPlane* bp=nullptr;
    if ( !rls.empty() ) {
      // First set the baseline plane to calculate relative poisions
      const auto& refSurf = (*rls.begin())->surface();
      if ( chid.region() == 0 ) {
        float corners[6] = {0,0,0,0,0,0};
        for ( auto rl : rls ) {
          const double h2 = rl->surface().bounds().length()/2;
          const double w2 = rl->surface().bounds().width()/2;
          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-w2,-h2,0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+w2,+h2,0)));
          corners[0] = std::min(corners[0], x1y1AtRef.x());
          corners[1] = std::min(corners[1], x1y1AtRef.y());
          corners[2] = std::max(corners[2], x2y2AtRef.x());
          corners[3] = std::max(corners[3], x2y2AtRef.y());

          corners[4] = std::min(corners[4], x1y1AtRef.z());
          corners[5] = std::max(corners[5], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((corners[0]+corners[2])/2, (corners[1]+corners[3])/2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new RectangularPlaneBounds((corners[2]-corners[0])/2, (corners[3]-corners[1])/2, (corners[5]-corners[4])+0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);
      }
      else {
        float cornersLo[3] = {0,0,0}, cornersHi[3] = {0,0,0};
        float cornersZ[2] = {0,0};
        for ( auto rl : rls ) {
          const double h2 = rl->surface().bounds().length()/2;
          const double w2 = rl->surface().bounds().width()/2;
          const auto& topo = dynamic_cast<const TrapezoidalStripTopology&>(rl->specificTopology());
          const double r = topo.radius();
          const double wAtLo = w2/r*(r-h2); // tan(theta/2) = (w/2)/r = x/(r-h/2)
          const double wAtHi = w2/r*(r+h2);

          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtLo, -h2, 0)));
          const auto x2y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtLo, -h2, 0)));
          const auto x1y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtHi, +h2, 0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtHi, +h2, 0)));

          cornersLo[0] = std::min(cornersLo[0], x1y1AtRef.x());
          cornersLo[1] = std::max(cornersLo[1], x2y1AtRef.x());
          cornersLo[2] = std::min(cornersLo[2], x1y1AtRef.y());

          cornersHi[0] = std::min(cornersHi[0], x1y2AtRef.x());
          cornersHi[1] = std::max(cornersHi[1], x2y2AtRef.x());
          cornersHi[2] = std::max(cornersHi[2], x1y2AtRef.y());

          cornersZ[0] = std::min(cornersZ[0], x1y1AtRef.z());
          cornersZ[1] = std::max(cornersZ[1], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((cornersHi[0]+cornersHi[1])/2, (cornersLo[2]+cornersHi[2])/2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new TrapezoidalPlaneBounds((cornersLo[1]-cornersLo[0])/2, (cornersHi[1]-cornersHi[0])/2, (cornersHi[2]-cornersLo[2])/2, (cornersZ[1]-cornersZ[0])+0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);
      }
    }

    ReferenceCountingPointer<BoundPlane> surf(bp);
    // Create the chamber
    RPCChamber* ch = new RPCChamber (chid, surf);
    // Add the rolls to rhe chamber
    for(auto rl : rls ) ch->add(rl);
    // Add the chamber to the geometry
    geometry->add(ch);

  }
  return geometry;
}

