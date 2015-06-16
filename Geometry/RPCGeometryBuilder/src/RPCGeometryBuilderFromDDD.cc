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
  std::string attribute = "ReadOutName"; // could come from .orcarc
  std::string value     = "MuonRPCHits";    // could come from .orcarc
  DDValue val(attribute, value, 0.0);

  // Asking only for the MuonRPC's
  DDSpecificsFilter filter;
  filter.setCriteria(val, // name & value of a variable 
		     DDCompOp::matches,
		     DDLogOp::AND, 
		     true, // compare strings otherwise doubles
		     true // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(*cview);
  fview.addFilter(filter);

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
    int detid = 0;

    LogDebug("RPCGeometryBuilderFromDDD") <<"Getting the Unit Number";
    detid = rpcnum.baseNumberToUnitNumber(mbn);
    LogDebug("RPCGeometryBuilderFromDDD") <<"Getting the RPC det Id "<<detid;

    RPCDetId rpcid(detid);
    RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);

    LogDebug("RPCGeometryBuilderFromDDD") <<"The RPCDetid is "<<rpcid;

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    std::vector<const DDsvalues_type* >::iterator is=specs.begin();
    int nStrips=0;
    for (;is!=specs.end(); is++){
      if (DDfetch( *is, numbOfStrips)){
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
//     Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
// 			      rota.yx(),rota.yy(),rota.yz(),
// 			      rota.zx(),rota.zy(),rota.zz());

//ROOT::Math way
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    // doesn't this just re-inverse???
    Surface::RotationType rot (float(x.X()),float(x.Y()),float(x.Z()),
			       float(y.X()),float(y.Y()),float(y.Z()),
			       float(z.X()),float(z.Y()),float(z.Z())); 
    
    std::vector<float> pars;
    RPCRollSpecs* rollspecs= 0;
    Bounds* bounds = 0;



    if (dpar.size()==3){
      float width     = dpar[0]/cm;
      float length    = dpar[1]/cm;
      float thickness = dpar[2]/cm;
      //RectangularPlaneBounds* 
      bounds = 
	new RectangularPlaneBounds(width,length,thickness);
      pars.push_back(width);
      pars.push_back(length);
      pars.push_back(numbOfStrips.doubles()[0]); //h/2;

      if (!theComp11Flag) {
	//Correction of the orientation to get the REAL geometry.
        //Change of axes for the +z part only.
        //Including the 0 whell
        if (tran.z() >-1500. ){
          Basic3DVector<float> newX(-1.,0.,0.);
          Basic3DVector<float> newY(0.,-1.,0.);
          Basic3DVector<float> newZ(0.,0.,1.);
          rot.rotateAxes (newX, newY,newZ);
        }
      }
      
      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel,name,pars);
      LogDebug("RPCGeometryBuilderFromDDD") <<"Barrel "<<name
					    <<" par "<<width
					    <<" "<<length<<" "<<thickness;
    }
    else{
      float be = dpar[4]/cm;
      float te = dpar[8]/cm;
      float ap = dpar[0]/cm;
      float ti = 0.4/cm;
      //  TrapezoidalPlaneBounds* 
      bounds = 
	new TrapezoidalPlaneBounds(be,te,ap,ti);
      pars.push_back(dpar[4]/cm); //b/2;
      pars.push_back(dpar[8]/cm); //B/2;
      pars.push_back(dpar[0]/cm); //h/2;
      pars.push_back(numbOfStrips.doubles()[0]); //h/2;
      
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

    std::list<RPCRoll *> rls;
    if (chids.find(chid)!=chids.end()){
      rls = chids[chid];
    }
    rls.push_back(r);
    chids[chid]=rls;

    doSubDets = fview.nextSibling(); // go to next layer
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
