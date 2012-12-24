/** \file
 *
 *  $Date: 2009/05/25 09:17:01 $
 *  $Revision: 1.13 $
 *  \author N. Amapane - CERN. 
 */

#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include <string>

using namespace std;

#include <string>

using namespace std;

DTGeometryBuilderFromDDD::DTGeometryBuilderFromDDD() {}

DTGeometryBuilderFromDDD::~DTGeometryBuilderFromDDD(){}


void DTGeometryBuilderFromDDD::build(boost::shared_ptr<DTGeometry> theGeometry,
                                     const DDCompactView* cview,
                                     const MuonDDDConstants& muonConstants){
  //  cout << "DTGeometryBuilderFromDDD::build" << endl;
  //   static const string t0 = "DTGeometryBuilderFromDDD::build";
  //   TimeMe timer(t0,true);

  std::string attribute = "MuStructure"; 
  std::string value     = "MuonBarrelDT";
  DDValue val(attribute, value, 0.0);

  // Asking only for the Muon DTs
  DDSpecificsFilter filter;
  filter.setCriteria(val,  // name & value of a variable 
		     DDSpecificsFilter::matches,
		     DDSpecificsFilter::AND, 
		     true, // compare strings otherwise doubles
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(*cview);
  fview.addFilter(filter);
  buildGeometry(theGeometry, fview, muonConstants);
}


void DTGeometryBuilderFromDDD::buildGeometry(boost::shared_ptr<DTGeometry> theGeometry,
                                             DDFilteredView& fv,
                                             const MuonDDDConstants& muonConstants) const {
  // static const string t0 = "DTGeometryBuilderFromDDD::buildGeometry";
  // TimeMe timer(t0,true);

  //DTGeometry* theGeometry = new DTGeometry;

  bool doChamber = fv.firstChild();

  // Loop on chambers
  int ChamCounter=0;
  while (doChamber){
    ChamCounter++;
    DDValue val("Type");
    const DDsvalues_type params(fv.mergedSpecifics());
    string type;
    if (DDfetch(&params,val)) type = val.strings()[0];
    // FIXME
    val=DDValue("FEPos");
    string FEPos;
    if (DDfetch(&params,val)) FEPos = val.strings()[0];
    DTChamber* chamber = buildChamber(fv,type, muonConstants);

    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter=0;
    while (doSL) {
      SLCounter++;
      DTSuperLayer* sl = buildSuperLayer(fv, chamber, type, muonConstants);
      theGeometry->add(sl);

      bool doL = fv.firstChild();
      int LCounter=0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        DTLayer* layer = buildLayer(fv, sl, type, muonConstants);
        theGeometry->add(layer);

        fv.parent();
        doL = fv.nextSibling(); // go to next layer
      } // layers

      fv.parent();
      doSL = fv.nextSibling(); // go to next SL
    } // sls
    theGeometry->add(chamber);

    fv.parent();
    doChamber = fv.nextSibling(); // go to next chamber
  } // chambers
}

DTChamber* DTGeometryBuilderFromDDD::buildChamber(DDFilteredView& fv,
                                                  const string& type, const MuonDDDConstants& muonConstants) const {
  MuonDDDNumbering mdddnum (muonConstants);
  DTNumberingScheme dtnum (muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTChamberId detId(rawid);  

  // Chamber specific parameter (size) 
  // FIXME: some trouble for boolean solids?
  vector<double> par = extractParameters(fv);

  float width = par[0]/cm;     // r-phi  dimension - different in different chambers
  float length = par[1]/cm;    // z      dimension - constant 125.55 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 18 cm

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z

  RCPPlane surf(plane(fv, new RectangularPlaneBounds(width, length, thickness) ));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer* DTGeometryBuilderFromDDD::buildSuperLayer(DDFilteredView& fv,
                                                        DTChamber* chamber,
                                                        const std::string& type, 
							const MuonDDDConstants& muonConstants) const {

  MuonDDDNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTSuperLayerId slId(rawid);

  // Slayer specific parameter (size)
  vector<double> par = extractParameters(fv);

  float width = par[0]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[1]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 20 cm

  // Ok this is the slayer position...
  RCPPlane surf(plane(fv, new RectangularPlaneBounds(width, length, thickness) ));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  //LocalPoint lpos(10,20,30);
  //GlobalPoint gpos=slayer->toGlobal(lpos);

  // add to the chamber
  chamber->add(slayer);
  return slayer;
}


DTLayer* DTGeometryBuilderFromDDD::buildLayer(DDFilteredView& fv,
                                              DTSuperLayer* sl,
                                              const std::string& type,
					      const MuonDDDConstants& muonConstants) const {

  MuonDDDNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);

  // Layer specific parameter (size)
  vector<double> par = extractParameters(fv);
  float width = par[0]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[1]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 20 cm

  RCPPlane surf(plane(fv, new RectangularPlaneBounds(width, length, thickness) ));

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter=0;
  int firstWire=fv.copyno();
  par = extractParameters(fv);
  float wireLength = par[1]/cm;
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling(); // next wire
  }
  //int lastWire=fv.copyno();
  DTTopology topology(firstWire, WCounter, wireLength);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);

  sl->add(layer);
  return layer;
}

vector<double> 
DTGeometryBuilderFromDDD::extractParameters(DDFilteredView& fv) const {
  vector<double> par;
  if (fv.logicalPart().solid().shape() != ddbox) {
    DDBooleanSolid bs(fv.logicalPart().solid());
    DDSolid A = bs.solidA();
    while (A.shape() != ddbox) {
      DDBooleanSolid bs(A);
      A = bs.solidA();
    }
    par=A.parameters();
  } else {
    par = fv.logicalPart().solid().parameters();
  }
  return par;
}

DTGeometryBuilderFromDDD::RCPPlane 
DTGeometryBuilderFromDDD::plane(const DDFilteredView& fv,
                                Bounds * bounds) const {
  // extract the position
  const DDTranslation & trans(fv.translation());

  const Surface::PositionType posResult(float(trans.x()/cm), 
                                        float(trans.y()/cm), 
                                        float(trans.z()/cm));
  // now the rotation
  //  DDRotationMatrix tmp = fv.rotation();
  // === DDD uses 'active' rotations - see CLHEP user guide ===
  //     ORCA uses 'passive' rotation. 
  //     'active' and 'passive' rotations are inverse to each other
  //  DDRotationMatrix tmp = fv.rotation();
  DDRotationMatrix rotation = fv.rotation();//REMOVED .Inverse();
  DD3Vector x, y, z;
  rotation.GetComponents(x,y,z);
//   std::cout << "INVERSE rotation by its own operator: "<< fv.rotation() << std::endl;
//   std::cout << "INVERSE rotation manually: "
// 	    << x.X() << ", " << x.Y() << ", " << x.Z() << std::endl
// 	    << y.X() << ", " << y.Y() << ", " << y.Z() << std::endl
// 	    << z.X() << ", " << z.Y() << ", " << z.Z() << std::endl;

  Surface::RotationType rotResult(float(x.X()),float(x.Y()),float(x.Z()),
                                  float(y.X()),float(y.Y()),float(y.Z()),
                                  float(z.X()),float(z.Y()),float(z.Z())); 

//   std::cout << "rotation by its own operator: "<< tmp << std::endl;
//   DD3Vector tx, ty,tz;
//   tmp.GetComponents(tx, ty, tz);
//   std::cout << "rotation manually: "
// 	    << tx.X() << ", " << tx.Y() << ", " << tx.Z() << std::endl
// 	    << ty.X() << ", " << ty.Y() << ", " << ty.Z() << std::endl
// 	    << tz.X() << ", " << tz.Y() << ", " << tz.Z() << std::endl;

  return RCPPlane( new Plane( posResult, rotResult, bounds));
}
