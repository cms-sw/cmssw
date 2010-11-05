/******* \class DTGeometryBuilderFromCondDB *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 19/11/2008 19:15:14 CET $
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromCondDB.h"

/* Collaborating Class Header */
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTGeometryBuilderFromCondDB::DTGeometryBuilderFromCondDB() {
}

/* Destructor */ 
DTGeometryBuilderFromCondDB::~DTGeometryBuilderFromCondDB() {
}

/* Operations */ 
void
DTGeometryBuilderFromCondDB::build(boost::shared_ptr<DTGeometry> theGeometry,
                                   const RecoIdealGeometry& rig) {
  //  cout << "DTGeometryBuilderFromCondDB " << endl;
  const std::vector<DetId>& detids(rig.detIds());
  //  cout << "size " << detids.size() << endl;

  size_t idt = 0;
  DTChamber* chamber(0);
  DTSuperLayer* sl(0);
  while(idt < detids.size()) {
    vector<double> par=rig.shapePars(idt);
    vector<double> gtran=rig.translation(idt);
    vector<double> grmat=rig.rotation(idt);
    //copy(par.begin(), par.end(), ostream_iterator<double>(std::cout," "));
    if (int(par[0])==0){ // a Chamber
      // add the provious chamber which by now has been updated with SL and
      // layers
      if (chamber) theGeometry->add(chamber);
      // go for the actual one
      DTChamberId chid(detids[idt]);
      //cout << "CH: " <<  chid << endl;
      chamber = buildChamber(chid, par, gtran, grmat);
    }
    else if (int(par[0])==1){ // a SL
      DTSuperLayerId slid(detids[idt]);
      //cout << "  SL: " <<  slid << endl;
      sl = buildSuperLayer(chamber, slid, par, gtran, grmat);
      theGeometry->add(sl);
    }
    else if (int(par[0])==2){ // a Layer
      DTLayerId lid(detids[idt]);
      //cout << "    LAY: " <<  lid << endl;
      DTLayer* lay = buildLayer(sl, lid, par, gtran, grmat);
      theGeometry->add(lay);
    } else { //what the fuck!!!
      cout << "What the Fuck is this!" << endl;
    }
    ++idt;
  }
  if (chamber) theGeometry->add(chamber); // add the last chamber
}

DTChamber* DTGeometryBuilderFromCondDB::buildChamber(const DetId& id,
                                                     const vector<double>& par,
                                                     const vector<double>& tran,
                                                     const vector<double>& rot) const {
  DTChamberId detId(id);  


  float width = par[1]/cm;     // r-phi  dimension - different in different chambers
  float length = par[2]/cm;    // z      dimension - constant 125.55 cm
  float thickness = par[3]/cm; // radial thickness - almost constant about 18 cm

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z
  RectangularPlaneBounds bound(width, length, thickness);

  RCPPlane surf(plane(tran, rot, bound));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer*
DTGeometryBuilderFromCondDB::buildSuperLayer(DTChamber* chamber,
                                             const DetId& id,
                                             const vector<double>& par,
                                             const vector<double>& tran,
                                             const vector<double>& rot) const {

  DTSuperLayerId slId(id);

  float width = par[1]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[2]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[3]/cm; // radial thickness - almost constant about 5 cm

  RectangularPlaneBounds bound(width, length, thickness);

  // Ok this is the slayer position...
  RCPPlane surf(plane(tran, rot ,bound));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  // cout << "adding slayer " << slayer->id() << " to chamber "<<  chamber->id() << endl;
  chamber->add(slayer);
  return slayer;
}

DTLayer*
DTGeometryBuilderFromCondDB::buildLayer(DTSuperLayer* sl,
                                        const DetId& id,
                                        const vector<double>& par,
                                        const vector<double>& tran,
                                        const vector<double>& rot) const {

  DTLayerId layId(id);

  // Layer specific parameter (size)
  float width = par[1]/cm;     // r-phi  dimension - changes in different chambers
  float length = par[2]/cm;    // z      dimension - constant 126.8 cm
  float thickness = par[3]/cm; // radial thickness - almost constant about 20 cm

  // define Bounds
  RectangularPlaneBounds bound(width, length, thickness);

  RCPPlane surf(plane(tran, rot, bound));

  // Loop on wires
  int firstWire=int(par[4]);
  int WCounter=int(par[5]);
  double sensibleLenght=par[6]/cm;
  DTTopology topology(firstWire, WCounter, sensibleLenght);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);
  // cout << "adding layer " << layer->id() << " to sl "<<  sl->id() << endl;

  sl->add(layer);
  return layer;
}

DTGeometryBuilderFromCondDB::RCPPlane 
DTGeometryBuilderFromCondDB::plane(const vector<double>& tran,
                                   const vector<double>& rot,
                                   const Bounds& bounds) const {
  // extract the position
  const Surface::PositionType posResult(tran[0], tran[1], tran[2]);
  // now the rotation
  Surface::RotationType rotResult( rot[0], rot[1], rot[2], 
                                   rot[3], rot[4], rot[5],
                                   rot[6], rot[7], rot[8] );

  return RCPPlane( new BoundPlane( posResult, rotResult, bounds));
}
