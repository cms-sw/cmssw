/** Implementation of the RPC Geometry Builder from DDD
 *
 *  \author Port of: MuDDDRPCBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

RPCGeometryBuilderFromDDD::RPCGeometryBuilderFromDDD(bool comp11) : theComp11Flag(comp11)
{ }

RPCGeometryBuilderFromDDD::~RPCGeometryBuilderFromDDD()
{ }

RPCGeometry* RPCGeometryBuilderFromDDD::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants)
{
  const std::string attribute = "ReadOutName"; // could come from .orcarc
  const std::string value     = "MuonRPCHits";    // could come from .orcarc

  // Asking only for the MuonRPC's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview,filter);

  return this->buildGeometry(fview, muonConstants);
}

RPCGeometry* RPCGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants)
{
  LogDebug("RPCGeometryBuilderFromDDD") <<"Building the geometry service";
  RPCGeometry* geometry = new RPCGeometry();

  LogDebug("RPCGeometryBuilderFromDDD") << "About to run through the RPC structure\n"
					<<" First logical part "
					<<fview.logicalPart().name().name();
  bool doSubDets = fview.firstChild();

  LogDebug("RPCGeometryBuilderFromDDD") << "doSubDets = " << doSubDets;
  while (doSubDets){
    LogDebug("RPCGeometryBuilderFromDDD") <<"start the loop";

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    LogDebug("RPCGeometryBuilderFromDDD") <<"Getting the Muon base Number";
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());
    LogDebug("RPCGeometryBuilderFromDDD") <<"Start the Rpc Numbering Schema";
    // Get the The Rpc det Id
    RPCNumberingScheme rpcnum(muonConstants);

    LogDebug("RPCGeometryBuilderFromDDD") <<"Getting the Unit Number";
    const int detid = rpcnum.baseNumberToUnitNumber(mbn);
    LogDebug("RPCGeometryBuilderFromDDD") <<"Getting the RPC det Id "<<detid;

    RPCDetId rpcid(detid);
    RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);

    LogDebug("RPCGeometryBuilderFromDDD") <<"The RPCDetid is "<<rpcid;

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    int nStrips=0;
    for (auto & spec : specs){
      if (DDfetch( spec, numbOfStrips)){
        nStrips=int(numbOfStrips.doubles()[0]);
      }
    }

    LogDebug("RPCGeometryBuilderFromDDD") << ((nStrips == 0 ) ? ("No strip found!!") : (""));

    std::vector<double> dpar=fview.logicalPart().solid().parameters();
    std::string name=fview.logicalPart().name().name();
    DDTranslation tran    = fview.translation();
    //removed .Inverse after comparing to DT...
    DDRotationMatrix rota = fview.rotation();//.Inverse();
    Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
    // CLHEP way
    // Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
    // 			                     rota.yx(),rota.yy(),rota.yz(),
    // 			                     rota.zx(),rota.zy(),rota.zz());

    //ROOT::Math way
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    // doesn't this just re-inverse???
    Surface::RotationType rot (float(x.X()),float(x.Y()),float(x.Z()),
                               float(y.X()),float(y.Y()),float(y.Z()),
                               float(z.X()),float(z.Y()),float(z.Z()));

    RPCRollSpecs* rollspecs= nullptr;
    Bounds* bounds = nullptr;

    if (dpar.size()==3){
      const float width     = dpar[0]/cm;
      const float length    = dpar[1]/cm;
      const float thickness = dpar[2]/cm;
      //RectangularPlaneBounds*
      bounds = new RectangularPlaneBounds(width,length,thickness);
      const std::vector<float> pars = {width, length, float(numbOfStrips.doubles()[0]) /*h/2*/};

      if (!theComp11Flag) {
        //Correction of the orientation to get the REAL geometry.
        //Change of axes for the +z part only.
        //Including the 0 wheel
        if (tran.z() >-1500. ){
          Basic3DVector<float> newX(-1.,0.,0.);
          Basic3DVector<float> newY(0.,-1.,0.);
          Basic3DVector<float> newZ(0.,0.,1.);
          rot.rotateAxes(newX, newY,newZ);
        }
      }

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel,name,pars);
      LogDebug("RPCGeometryBuilderFromDDD") <<"Barrel "<<name
					    <<" par "<<width
					    <<" "<<length<<" "<<thickness;
    }
    else {
      const float be = dpar[4]/cm;
      const float te = dpar[8]/cm;
      const float ap = dpar[0]/cm;
      const float ti = 0.4/cm;
      //  TrapezoidalPlaneBounds*
      bounds = new TrapezoidalPlaneBounds(be,te,ap,ti);
      const std::vector<float> pars = {float(dpar[4]/cm) /*b/2*/, float(dpar[8]/cm) /*B/2*/,
                                       float(dpar[0]/cm) /*h/2*/, float(numbOfStrips.doubles()[0]) /*h/2*/};

      LogDebug("RPCGeometryBuilderFromDDD") <<"Forward "<<name
					    <<" par "<<dpar[4]/cm
					    <<" "<<dpar[8]/cm<<" "<<dpar[3]/cm<<" "
					    <<dpar[0];

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap,name,pars);

      //Change of axes for the forward
      Basic3DVector<float> newX(1.,0.,0.);
      Basic3DVector<float> newY(0.,0.,1.);
      //      if (tran.z() > 0. )
      newY *= -1;
      Basic3DVector<float> newZ(0.,1.,0.);
      rot.rotateAxes (newX, newY,newZ);
    }
    LogDebug("RPCGeometryBuilderFromDDD") <<"   Number of strips "<<nStrips;

    BoundPlane* bp = new BoundPlane(pos,rot,bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    RPCRoll* r=new RPCRoll(rpcid,surf,rollspecs);
    geometry->add(r);

    auto rls = chids.find(chid);
    if ( rls == chids.end() ) rls = chids.insert(std::make_pair(chid, std::list<RPCRoll*>())).first;
    rls->second.emplace_back(r);

    doSubDets = fview.nextSibling(); // go to next layer
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
    for ( auto rl : rls ) ch->add(rl);
    // Add the chamber to the geometry
    geometry->add(ch);

  }
  return geometry;
}
