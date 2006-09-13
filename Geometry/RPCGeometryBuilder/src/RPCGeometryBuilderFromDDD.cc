/** Implementation of the RPC Geometry Builder from DDD
 *
 *  \author Port of: MuDDDRPCBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"

#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"

#include "Geometry/Vector/interface/Basic3DVector.h"

#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <algorithm>

RPCGeometryBuilderFromDDD::RPCGeometryBuilderFromDDD()
{ }

RPCGeometryBuilderFromDDD::~RPCGeometryBuilderFromDDD() 
{ }

RPCGeometry* RPCGeometryBuilderFromDDD::build(const DDCompactView* cview)
{

  try {

    std::string attribute = "ReadOutName"; // could come from .orcarc
    std::string value     = "MuonRPCHits";    // could come from .orcarc
    DDValue val(attribute, value, 0.0);

    // Asking only for the MuonRPC's
    DDSpecificsFilter filter;
    filter.setCriteria(val, // name & value of a variable 
		       DDSpecificsFilter::matches,
		       DDSpecificsFilter::AND, 
		       true, // compare strings otherwise doubles
		       true // use merged-specifics or simple-specifics
		       );
    DDFilteredView fview(*cview);
    fview.addFilter(filter);

    return buildGeometry(fview);
  }
  catch (const DDException & e ) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"DDD Exception: something went wrong during XML parsing!" 
	      << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const cms::Exception& e){  
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"an unexpected exception occured: " 
	      << e << std::endl;   
    throw;
  }
  catch (const std::exception& e) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"an unexpected exception occured: " 
	      << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}

RPCGeometry* RPCGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview) const
{

  RPCGeometry* geometry = new RPCGeometry();
  
  std::cout << "About to run through the RPC structure" << std::endl;
  std::cout <<" First logical part "
  	    <<fview.logicalPart().name().name()<<std::endl;

  bool doChamber = fview.firstChild();

  //-----------------DEBUG----------------------------------
  std::cout << "doChamber is "<<doChamber<< std::endl;
  //--------------------------------------------------------

  int ChamCounter = 0;
  while (doChamber){

    ChamCounter++;


    RPCChamber* chamber = buildChamber(fview);

    bool doRoll = fview.firstChild();
    std::cout <<"doRoll is " <<doRoll<<std::endl;
    int rollCounter = 0;
    while(doRoll){
      rollCounter++;
      std::cout <<"Built  "<< rollCounter << "  RPCRoll!" <<std::endl;
      RPCRoll* r = buildRoll(fview,chamber);
      geometry->add(r);
      fview.parent();
      doRoll = fview.nextSibling();
    }
    geometry->add(chamber);
    fview.parent();
    doChamber = fview.nextSibling(); // go to next chamber
  } // chambers
  std::cout <<"Built  "<< ChamCounter << "  RPCChamber!" <<std::endl;
  return geometry;
}

RPCChamber* RPCGeometryBuilderFromDDD::buildChamber(DDFilteredView& fview) const
{
  // Get the Base Muon Number
  MuonDDDNumbering mdddnum;
  MuonBaseNumber mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

  // Get the The Rpc det Id 
  RPCNumberingScheme rpcnum;
  int detid = 0;
  detid = rpcnum.baseNumberToUnitNumber(mbn);

  RPCDetId rpcid(detid);

  DDValue numbOfStrips("nStrips");
  std::vector<const DDsvalues_type* > specs(fview.specifics());
  std::vector<const DDsvalues_type* >::iterator is=specs.begin();
  int nStrips=0;

  for (;is!=specs.end(); is++){
    if (DDfetch( *is, numbOfStrips)){
      nStrips=int(numbOfStrips.doubles()[0]);	
    }
  }

  if (nStrips == 0 )
    std::cout <<"No strip found!!"<<std::endl;

  std::vector<double> dpar=fview.logicalPart().solid().parameters();
//   std::string name=fview.logicalPart().name().name();
  const DDTranslation& tran(fview.translation());
  DDRotationMatrix rota = fview.rotation().inverse();
  Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
  Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
			    rota.yx(),rota.yy(),rota.yz(),
			    rota.zx(),rota.zy(),rota.zz());

//   std::vector<float> pars;
//   RPCRollSpecs* rollspecs= 0;
  Bounds* bound = 0;
  
  if (dpar.size()==3){
    float width     = dpar[0]/cm;
    float length    = dpar[1]/cm;
    float thickness = dpar[2]/cm;
    
    //RectangularPlaneBounds* 
    bound = new RectangularPlaneBounds(width, length, thickness);
//     pars.push_back(width);
//     pars.push_back(length);
//     pars.push_back(numbOfStrips.doubles()[0]); //h/2;
//     rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel,name,pars);

    std::cout <<"Barrel "<<fview.logicalPart().name().name()
	      <<" par "<<width
	      <<" "<<length<<" "<<thickness;
    
  }
  else{
    float be = dpar[4]/cm;
    float te = dpar[8]/cm;
    float ap = dpar[0]/cm;
    float ti = 0.4/cm;
    
    //  TrapezoidalPlaneBounds* 
    bound = new TrapezoidalPlaneBounds(be,te,ap,ti);
//     pars.push_back(dpar[4]/cm); //b/2;
//     pars.push_back(dpar[8]/cm); //B/2;
//     pars.push_back(dpar[0]/cm); //h/2;
//     pars.push_back(numbOfStrips.doubles()[0]); //h/2;
//     rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap,name,pars);

    std::cout <<"Forward "<<fview.logicalPart().name().name()
	      <<" par "<<dpar[4]/cm
	      <<" "<<dpar[8]/cm<<" "<<dpar[3]/cm<<" "
	      <<dpar[0];
    
    //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);

    if (tran.z() > 0. ) newY *= -1;

    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY,newZ);
    
  }
  
  std::cout <<"   Number of strips "<<nStrips<<std::endl;
  
  BoundPlane::BoundPlanePointer bp = BoundPlane::build(pos,rot,bound);
  delete bound;
  std::cout <<"   Bound Plane OK! "<<std::endl;
  RPCChamber* ch = new RPCChamber(rpcid,bp);

  std::cout <<"   Return Chamber "<<std::endl;

  return ch;
}

RPCRoll* RPCGeometryBuilderFromDDD::buildRoll(DDFilteredView& fview,
					      RPCChamber* ch) const
{
  // Get the Base Muon Number
  MuonDDDNumbering mdddnum;
  MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

  // Get the The Rpc det Id 
  RPCNumberingScheme rpcnum;
  int detid = 0;
  detid = rpcnum.baseNumberToUnitNumber(mbn);

  RPCDetId rpcid(detid);
  DDValue numbOfStrips("nStrips");
  
  std::vector<const DDsvalues_type* > specs(fview.specifics());
  std::vector<const DDsvalues_type* >::iterator is=specs.begin();
  int nStrips=0;

  for (;is!=specs.end(); is++){
    if (DDfetch( *is, numbOfStrips)){
      nStrips=int(numbOfStrips.doubles()[0]);	
    }
  }

  if (nStrips == 0 )
    std::cout <<"No strip found!!"<<std::endl;
  
  std::vector<double> dpar=fview.logicalPart().solid().parameters();
  std::string name=fview.logicalPart().name().name();
  DDTranslation tran    = fview.translation();
  DDRotationMatrix rota = fview.rotation().inverse();
  Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
  Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
			    rota.yx(),rota.yy(),rota.yz(),
			    rota.zx(),rota.zy(),rota.zz());
  std::vector<float> pars;
  RPCRollSpecs* rollspecs= 0;
  Bounds* bound = 0;
  
  if (dpar.size()==3){
    float width     = dpar[0]/cm;
    float length    = dpar[1]/cm;
    float thickness = dpar[2]/cm;
    
    //RectangularPlaneBounds* 
    bound = new RectangularPlaneBounds(width, length, thickness);
    pars.push_back(width);
    pars.push_back(length);
    pars.push_back(numbOfStrips.doubles()[0]); //h/2;
    rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel,name,pars);
    
    std::cout <<"Barrel "<<name
	      <<" par "<<width
	      <<" "<<length<<" "<<thickness;
    
  }
  else{
    float be = dpar[4]/cm;
    float te = dpar[8]/cm;
    float ap = dpar[0]/cm;
    float ti = 0.4/cm;
    
    //  TrapezoidalPlaneBounds* 
    bound = new TrapezoidalPlaneBounds(be,te,ap,ti);
    
    pars.push_back(dpar[4]/cm); //b/2;
    pars.push_back(dpar[8]/cm); //B/2;
    pars.push_back(dpar[0]/cm); //h/2;
    pars.push_back(numbOfStrips.doubles()[0]); //h/2;
    
    std::cout <<"Forward "<<name
	      <<" par "<<dpar[4]/cm
	      <<" "<<dpar[8]/cm<<" "<<dpar[3]/cm<<" "
	      <<dpar[0];
    
    rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap,name,pars);
    
    //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);

    if (tran.z() > 0. ) newY *= -1;

    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY,newZ);
    
  }
  
  std::cout <<"   Number of strips "<<nStrips<<std::endl;
  
  BoundPlane::BoundPlanePointer bp = BoundPlane::build(pos,rot,bound);
  delete bound;
  RPCRoll* r=new RPCRoll(rpcid,bp,rollspecs,ch); //specs ownership is given to the roll
  ch->add(r);
  
  return r;
}

// std::vector<double> 
// RPCGeometryBuilderFromDDD::extractParameters(DDFilteredView& fv) const {
//   std::vector<double> par;
//   if (fv.logicalPart().solid().shape() != ddbox) {
//     DDBooleanSolid bs(fv.logicalPart().solid());
//     DDSolid A = bs.solidA();
//     while (A.shape() != ddbox) {
//       DDBooleanSolid bs(A);
//       A = bs.solidA();
//     }
//     par=A.parameters();
//   } else {
//     par = fv.logicalPart().solid().parameters();
//   }
//   return par;
// }

// RPCGeometryBuilderFromDDD::RCPPlane 
// RPCGeometryBuilderFromDDD::plane(const DDFilteredView& fv,
//                                 const Bounds& bounds) const {
//   // extract the position
//   const DDTranslation & trans(fv.translation());

//   const Surface::PositionType posResult(float(trans.x()/cm), 
//                                         float(trans.y()/cm), 
//                                         float(trans.z()/cm));
//   // now the rotation
//   DDRotationMatrix tmp = fv.rotation();
//   // === DDD uses 'active' rotations - see CLHEP user guide ===
//   //     ORCA uses 'passive' rotation. 
//   //     'active' and 'passive' rotations are inverse to each other
//   DDRotationMatrix rotation = tmp.inverse();

//   Surface::RotationType rotResult(float(rotation.xx()),float(rotation.xy()),float(rotation.xz()),
//                                   float(rotation.yx()),float(rotation.yy()),float(rotation.yz()),
//                                   float(rotation.zx()),float(rotation.zy()),float(rotation.zz())); 

//   return RCPPlane( new BoundPlane( posResult, rotResult, bounds));
// }
