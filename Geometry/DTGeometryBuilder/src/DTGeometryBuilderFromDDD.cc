/** \file
 *
 *  $Date: 2006/04/27 11:01:30 $
 *  $Revision: 1.3 $
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
#include "CLHEP/Units/SystemOfUnits.h"


#include "Geometry/Surface/interface/RectangularPlaneBounds.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/Surface/interface/RectangularPlaneBounds.h"


#include <string>

using namespace std;

#include <string>

using namespace std;

DTGeometryBuilderFromDDD::DTGeometryBuilderFromDDD() {}

DTGeometryBuilderFromDDD::~DTGeometryBuilderFromDDD(){}


DTGeometry* DTGeometryBuilderFromDDD::build(const DDCompactView* cview){
  //   static const string t0 = "DTGeometryBuilderFromDDD::build";
  //   TimeMe timer(t0,true);

  try {
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

    return buildGeometry(fview);
  }
  catch (const DDException & e ) {
    std::cerr << "DTGeometryBuilderFromDDD::build() : DDD Exception: something went wrong during XML parsing!" << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const exception & e) {
    std::cerr << "DTGeometryBuilderFromDDD::build() : an unexpected exception occured: " << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr << "DTGeometryBuilderFromDDD::build() : An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}


DTGeometry* DTGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fv) const {
  // static const string t0 = "DTGeometryBuilderFromDDD::buildGeometry";
  // TimeMe timer(t0,true);

  DTGeometry* theGeometry = new DTGeometry;

  cout << "Starting from " << fv.logicalPart().name().name() << endl;

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
    DTChamber* chamber = buildChamber(fv,type);

    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter=0;
    while (doSL) {
      SLCounter++;
      DTSuperLayer* sl = buildSuperLayer(fv, chamber, type);
      theGeometry->add(sl);

      bool doL = fv.firstChild();
      int LCounter=0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        DTLayer* layer = buildLayer(fv, sl, type);
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
  cout << "Built " << ChamCounter << " drift tube chambers." << endl;
  return theGeometry;
}

DTChamber* DTGeometryBuilderFromDDD::buildChamber(DDFilteredView& fv,
                                                  const string& type) const {
  MuonDDDNumbering mdddnum;
  DTNumberingScheme dtnum;
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTChamberId detId(rawid);  

  // cout << " DTDetBuilder::buildChamber type: " << type
  //   << " id: " << detId << endl;  
  // fv.print();
  // cout << "Category " << int(fv.logicalPart().category()) << endl;
  // cout << "Shape " << int(fv.logicalPart().solid().shape()) << endl;

  // Chamber specific parameter (size) 
  // FIXME: some trouble for boolean solids?
  vector<double> par = extractParameters(fv);

  float width = par[0]/cm;     // r-phi  dimension - different in different chambers
  float length = par[1]/cm;    // z      dimension - constant 125.55 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 18 cm

  //cout << "width " << width << " length " << length << " thickness " << thickness << endl;

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z
  RectangularPlaneBounds bound(width, length, thickness);

  RCPPlane surf(plane(fv,bound));

  DTChamber* chamber = new DTChamber(detId, surf);
  //cout << "ChamberSurf " << &(chamber->surface())<< endl;

  return chamber;
}

DTSuperLayer* DTGeometryBuilderFromDDD::buildSuperLayer(DDFilteredView& fv,
                                                        DTChamber* chamber,
                                                        const std::string& type) const {

  MuonDDDNumbering mdddnum;
  DTNumberingScheme dtnum;
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTSuperLayerId slId(rawid);

  //cout << " DTDetBuilder::buildSuperLayer type: " << type << " id: " << slId << endl;  
  // Slayer specific parameter (size)
  vector<double> par = extractParameters(fv);

  float width = par[0]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[1]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 20 cm
  //cout << "width " << width << " length " << length << " thickness " << thickness << endl;

  RectangularPlaneBounds bound(width, length, thickness);
  // cout << "SL: width " << bound.width() << " length " << bound.length() << 
  //   " thickness " << bound.thickness() << endl;

  // Ok this is the slayer position...
  RCPPlane surf(plane(fv,bound));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  //LocalPoint lpos(10,20,30);
  //GlobalPoint gpos=slayer->toGlobal(lpos);
  // cout << "Chamber global pos " << chamber->position() << endl;
  // cout << "SLayer  global pos " << slayer->position()  << endl;
  // cout << "SLayer  local  pos " << chamber->toLocal(slayer->position()) << endl;
  // cout << "Local pos " << lpos << " global " << (gpos-slayer->position()) << endl;

  // add to the chamber
  chamber->add(slayer);
  return slayer;
}


DTLayer* DTGeometryBuilderFromDDD::buildLayer(DDFilteredView& fv,
                                              DTSuperLayer* sl,
                                              const std::string& type) const {

  MuonDDDNumbering mdddnum;
  DTNumberingScheme dtnum;
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);

  //cout << " DTDetBuilder::buildLayer type: " << type << " id: " << layId << endl;  

  // Layer specific parameter (size)
  vector<double> par = extractParameters(fv);
  float width = par[0]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[1]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[2]/cm; // radial thickness - almost constant about 20 cm

  // define Bounds
  RectangularPlaneBounds bound(width, length, thickness);

  RCPPlane surf(plane(fv,bound));

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter=0;
  int firstWire=fv.copyno();
  par = extractParameters(fv);
  float wireLength = par[1]/cm;
//   cout << " first wire: " << firstWire << " " << fv.logicalPart().name().name();
//   cout << " lenght: " << wireLength << " layer lenght: " << length;
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling(); // next wire
  }
  //int lastWire=fv.copyno();
  // cout << " last wire: " << fv.copyno()
  //   << " total: " << WCounter << endl;
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
                                const Bounds& bounds) const {
  // extract the position
  const DDTranslation & trans(fv.translation());

  const Surface::PositionType posResult(float(trans.x()/cm), 
                                        float(trans.y()/cm), 
                                        float(trans.z()/cm));
  // now the rotation
  DDRotationMatrix tmp = fv.rotation();
  // === DDD uses 'active' rotations - see CLHEP user guide ===
  //     ORCA uses 'passive' rotation. 
  //     'active' and 'passive' rotations are inverse to each other
  DDRotationMatrix rotation = tmp.inverse();

  Surface::RotationType rotResult(float(rotation.xx()),float(rotation.xy()),float(rotation.xz()),
                                  float(rotation.yx()),float(rotation.yy()),float(rotation.yz()),
                                  float(rotation.zx()),float(rotation.zy()),float(rotation.zz())); 

  return RCPPlane( new BoundPlane( posResult, rotResult, bounds));
}
