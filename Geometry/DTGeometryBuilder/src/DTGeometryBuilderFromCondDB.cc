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
#include "DataFormats/Math/interface/GeantUnits.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

/* C++ Headers */
#include <iostream>
using namespace std;

using namespace geant_units;
using namespace geant_units::operators;

//#define EDM_ML_DEBUG
/* ====================================================================== */

/* Constructor */
DTGeometryBuilderFromCondDB::DTGeometryBuilderFromCondDB() {}

/* Destructor */
DTGeometryBuilderFromCondDB::~DTGeometryBuilderFromCondDB() {}

/* Operations */
void DTGeometryBuilderFromCondDB::build(const std::shared_ptr<DTGeometry>& theGeometry, const RecoIdealGeometry& rig) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "DTGeometryBuilderFromCondDB ";
#endif
  const std::vector<DetId>& detids(rig.detIds());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "size " << detids.size();
#endif

  size_t idt = 0;
  DTChamber* chamber(nullptr);
  DTSuperLayer* sl(nullptr);
  while (idt < detids.size()) {
    //copy(par.begin(), par.end(), ostream_iterator<double>(std::cout," "));
    if (int(*(rig.shapeStart(idt))) == 0) {  // a Chamber
      // add the provious chamber which by now has been updated with SL and
      // layers
      if (chamber)
        theGeometry->add(chamber);
      // go for the actual one
      DTChamberId chid(detids[idt]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DTGeometry") << "CH: " <<  chid;
#endif
      chamber = buildChamber(chid, rig, idt);
    } else if (int(*(rig.shapeStart(idt))) == 1) {  // a SL
      DTSuperLayerId slid(detids[idt]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DTGeometry") << "  SL: " <<  slid;
#endif
      sl = buildSuperLayer(chamber, slid, rig, idt);
      theGeometry->add(sl);
    } else if (int(*(rig.shapeStart(idt))) == 2) {  // a Layer
      DTLayerId lid(detids[idt]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DTGeometry") << "    LAY: " <<  lid;
#endif
      DTLayer* lay = buildLayer(sl, lid, rig, idt);
      theGeometry->add(lay);
    } else {
      edm::LogVerbatim("DTGeometry") << "What is this?";
    }
    ++idt;
  }
  if (chamber)
    theGeometry->add(chamber);  // add the last chamber
}

// Calling function has the responsibility to delete the allocated RectangularPlaneBounds object
RectangularPlaneBounds* dtGeometryBuilder::getRecPlaneBounds(const std::vector<double>::const_iterator& shapeStart) {
  float width = convertMmToCm(*(shapeStart));          // r-phi  dimension - different in different chambers
  float length = convertMmToCm(*(shapeStart + 1));     // z      dimension - constant
  float thickness = convertMmToCm(*(shapeStart + 2));  // radial thickness - almost constant
  return new RectangularPlaneBounds(width, length, thickness);
}

DTChamber* DTGeometryBuilderFromCondDB::buildChamber(const DetId& id, const RecoIdealGeometry& rig, size_t idt) const {
  DTChamberId detId(id);

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // length z      dimension - constant 125.55 cm
  // thickness is along local Z
  // radial thickness - almost constant about 18 cm
  RCPPlane surf(
      plane(rig.tranStart(idt), rig.rotStart(idt), dtGeometryBuilder::getRecPlaneBounds(++rig.shapeStart(idt))));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer* DTGeometryBuilderFromCondDB::buildSuperLayer(DTChamber* chamber,
                                                           const DetId& id,
                                                           const RecoIdealGeometry& rig,
                                                           size_t idt) const {
  DTSuperLayerId slId(id);

  // r-phi  dimension - different in different chambers
  // z      dimension - constant 126.8 cm
  // radial thickness - almost constant about 5 cm

  // Ok this is the slayer position...
  RCPPlane surf(
      plane(rig.tranStart(idt), rig.rotStart(idt), dtGeometryBuilder::getRecPlaneBounds(++rig.shapeStart(idt))));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "adding slayer " << slayer->id() << " to chamber "<<  chamber->id();
#endif
  assert(chamber);
  chamber->add(slayer);
  return slayer;
}

DTLayer* DTGeometryBuilderFromCondDB::buildLayer(DTSuperLayer* sl,
                                                 const DetId& id,
                                                 const RecoIdealGeometry& rig,
                                                 size_t idt) const {
  DTLayerId layId(id);

  // Layer specific parameter (size)
  // r-phi  dimension - different in different chambers
  // z      dimension - constant 126.8 cm
  // radial thickness - almost constant about 20 cm

  auto shapeStartPtr = rig.shapeStart(idt);
  RCPPlane surf(
      plane(rig.tranStart(idt), rig.rotStart(idt), dtGeometryBuilder::getRecPlaneBounds((shapeStartPtr + 1))));

  // Loop on wires
  int firstWire = static_cast<int>(*(shapeStartPtr + 4));       //par[4]);
  int WCounter = static_cast<int>(*(shapeStartPtr + 5));        //par[5]);
  double sensibleLength = convertMmToCm(*(shapeStartPtr + 6));  //par[6] in cm;
  DTTopology topology(firstWire, WCounter, sensibleLength);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "adding layer " << layer->id() << " to sl "<<  sl->id();
#endif
  assert(sl);
  sl->add(layer);
  return layer;
}

DTGeometryBuilderFromCondDB::RCPPlane DTGeometryBuilderFromCondDB::plane(const vector<double>::const_iterator tranStart,
                                                                         const vector<double>::const_iterator rotStart,
                                                                         Bounds* bounds) const {
  // extract the position
  const Surface::PositionType posResult(*(tranStart), *(tranStart + 1), *(tranStart + 2));
  // now the rotation
  Surface::RotationType rotResult(*(rotStart + 0),
                                  *(rotStart + 1),
                                  *(rotStart + 2),
                                  *(rotStart + 3),
                                  *(rotStart + 4),
                                  *(rotStart + 5),
                                  *(rotStart + 6),
                                  *(rotStart + 7),
                                  *(rotStart + 8));

  return RCPPlane(new Plane(posResult, rotResult, bounds));
}
