/******* \class DTGeometryBuilderFromCondDB *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
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
DTGeometryBuilderFromCondDB::build(const std::shared_ptr<DTGeometry>& theGeometry,
                                   const RecoIdealGeometry& rig) {
  //  cout << "DTGeometryBuilderFromCondDB " << endl;
  const std::vector<DetId>& detids(rig.detIds());
  //  cout << "size " << detids.size() << endl;

  size_t idt = 0;
  DTChamber* chamber(nullptr);
  DTSuperLayer* sl(nullptr);
  while(idt < detids.size()) {
    //copy(par.begin(), par.end(), ostream_iterator<double>(std::cout," "));
    if (int(*(rig.shapeStart(idt)))==0){ // a Chamber
      // add the provious chamber which by now has been updated with SL and
      // layers
      if (chamber) theGeometry->add(chamber);
      // go for the actual one
      DTChamberId chid(detids[idt]);
      //cout << "CH: " <<  chid << endl;
      chamber = buildChamber(chid, rig, idt); 
    }
    else if (int(*(rig.shapeStart(idt)))==1){ // a SL
      DTSuperLayerId slid(detids[idt]);
      //cout << "  SL: " <<  slid << endl;
      sl = buildSuperLayer(chamber, slid, rig, idt);
      theGeometry->add(sl);
    }
    else if (int(*(rig.shapeStart(idt)))==2){ // a Layer
      DTLayerId lid(detids[idt]);
      //cout << "    LAY: " <<  lid << endl;
      DTLayer* lay = buildLayer(sl, lid, rig, idt);
      theGeometry->add(lay);
    } else {
      cout << "What is this?" << endl;
    }
    ++idt;
  }
  if (chamber) theGeometry->add(chamber); // add the last chamber
}

DTChamber* DTGeometryBuilderFromCondDB::buildChamber(const DetId& id,
                                                     const RecoIdealGeometry& rig,
						     size_t idt ) const {
  DTChamberId detId(id);  


  float width = (*(rig.shapeStart(idt) + 1))/cm;     // r-phi  dimension - different in different chambers
  float length = (*(rig.shapeStart(idt) + 2))/cm;    // z      dimension - constant 125.55 cm
  float thickness = (*(rig.shapeStart(idt) + 3))/cm; // radial thickness - almost constant about 18 cm

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z
  RCPPlane surf(plane(rig.tranStart(idt), rig.rotStart(idt), new RectangularPlaneBounds(width, length, thickness) ));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer*
DTGeometryBuilderFromCondDB::buildSuperLayer(DTChamber* chamber,
                                             const DetId& id,
                                             const RecoIdealGeometry& rig,
					     size_t idt) const {

  DTSuperLayerId slId(id);

  float width = (*(rig.shapeStart(idt) + 1))/cm;     // r-phi  dimension - different in different chambers
  float length = (*(rig.shapeStart(idt) + 2))/cm;    // z      dimension - constant 126.8 cm
  float thickness = (*(rig.shapeStart(idt) + 3))/cm; // radial thickness - almost constant about 5 cm

  // Ok this is the slayer position...
  RCPPlane surf(plane(rig.tranStart(idt), rig.rotStart(idt), new RectangularPlaneBounds(width, length, thickness) ));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  // cout << "adding slayer " << slayer->id() << " to chamber "<<  chamber->id() << endl;
  assert(chamber);
  chamber->add(slayer);
  return slayer;
}

DTLayer*
DTGeometryBuilderFromCondDB::buildLayer(DTSuperLayer* sl,
                                        const DetId& id,
                                        const RecoIdealGeometry& rig,
					size_t idt) const {

  DTLayerId layId(id);

  // Layer specific parameter (size)
  float width = (*(rig.shapeStart(idt) + 1))/cm;     // r-phi  dimension - changes in different chambers
  float length = (*(rig.shapeStart(idt) + 2))/cm;    // z      dimension - constant 126.8 cm
  float thickness = (*(rig.shapeStart(idt) + 3))/cm; // radial thickness - almost constant about 20 cm


  RCPPlane surf(plane(rig.tranStart(idt), rig.rotStart(idt), new RectangularPlaneBounds(width, length, thickness) ));

  // Loop on wires
  int firstWire=int(*(rig.shapeStart(idt) + 4 ));//par[4]);
  int WCounter=int(*(rig.shapeStart(idt) + 5 ));//par[5]);
  double sensibleLenght=(*(rig.shapeStart(idt) + 6 ))/cm;//par[6]/cm;
  DTTopology topology(firstWire, WCounter, sensibleLenght);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);
  // cout << "adding layer " << layer->id() << " to sl "<<  sl->id() << endl;

  assert(sl);
  sl->add(layer);
  return layer;
}

DTGeometryBuilderFromCondDB::RCPPlane 
DTGeometryBuilderFromCondDB::plane(const vector<double>::const_iterator tranStart,
                                   const vector<double>::const_iterator rotStart,
                                   Bounds * bounds) const {
  // extract the position
  const Surface::PositionType posResult(*(tranStart), *(tranStart+1), *(tranStart+2));
  // now the rotation
  Surface::RotationType rotResult( *(rotStart+0), *(rotStart+1), *(rotStart+2), 
                                   *(rotStart+3), *(rotStart+4), *(rotStart+5),
                                   *(rotStart+6), *(rotStart+7), *(rotStart+8) );

  return RCPPlane( new Plane( posResult, rotResult, bounds));
}
