/** \file
 *
 *  \author N. Amapane - CERN. 
 */

#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "DataFormats/Math/interface/GeantUnits.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <utility>

using namespace std;
using namespace geant_units;
using namespace geant_units::operators;
using namespace std;

//#define EDM_ML_DEBUG

DTGeometryBuilderFromDDD::DTGeometryBuilderFromDDD() {}

DTGeometryBuilderFromDDD::~DTGeometryBuilderFromDDD() {}

void DTGeometryBuilderFromDDD::build(DTGeometry& theGeometry,
                                     const DDCompactView* cview,
                                     const MuonGeometryConstants& muonConstants) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "DTGeometryBuilderFromDDD::build";
  //static const string t0 = "DTGeometryBuilderFromDDD::build";
  //TimeMe timer(t0,true);
#endif

  std::string attribute = "MuStructure";
  std::string value = "MuonBarrelDT";

  // Asking only for the Muon DTs
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};

  DDFilteredView fview(*cview, filter);
  buildGeometry(theGeometry, fview, muonConstants);
}

void DTGeometryBuilderFromDDD::buildGeometry(DTGeometry& theGeometry,
                                             DDFilteredView& fv,
                                             const MuonGeometryConstants& muonConstants) const {
  edm::LogVerbatim("DTGeometryBuilder") << "(0) DTGeometryBuilder - DDD ";

  bool doChamber = fv.firstChild();

  // Loop on chambers
  int ChamCounter = 0;
  while (doChamber) {
    ChamCounter++;
    DDValue val("Type");
    const DDsvalues_type params(fv.mergedSpecifics());
    string type;
    if (DDfetch(&params, val))
      type = val.strings()[0];
    // FIXME
    val = DDValue("FEPos");
    string FEPos;
    if (DDfetch(&params, val))
      FEPos = val.strings()[0];
    DTChamber* chamber = buildChamber(fv, type, muonConstants);

    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter = 0;
    while (doSL) {
      SLCounter++;
      DTSuperLayer* sl = buildSuperLayer(fv, chamber, type, muonConstants);
      theGeometry.add(sl);

      bool doL = fv.firstChild();
      int LCounter = 0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        DTLayer* layer = buildLayer(fv, sl, type, muonConstants);
        theGeometry.add(layer);

        fv.parent();
        doL = fv.nextSibling();  // go to next layer
      }                          // layers

      fv.parent();
      doSL = fv.nextSibling();  // go to next SL
    }                           // sls
    theGeometry.add(chamber);

    fv.parent();
    doChamber = fv.nextSibling();  // go to next chamber
  }                                // chambers
}

DTChamber* DTGeometryBuilderFromDDD::buildChamber(DDFilteredView& fv,
                                                  const string& type,
                                                  const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTChamberId detId(rawid);

  // Chamber specific parameter (size)
  // FIXME: some trouble for boolean solids?
  std::vector<double> par = extractParameters(fv);

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X. r-phi  dimension - different in different chambers
  // length is along local Y. z      dimension - constant 125.55 cm
  // thickness is long local Z. radial thickness - almost constant about 18 cm

  edm::LogVerbatim("DTGeometryBuilder") << "(1) detId: " << rawid << " par[0]: " << par[0] << " par[1]: " << par[1]
                                        << " par[2]: " << par[2];

  RCPPlane surf(plane(fv, dtGeometryBuilder::getRecPlaneBounds(par.begin())));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer* DTGeometryBuilderFromDDD::buildSuperLayer(DDFilteredView& fv,
                                                        DTChamber* chamber,
                                                        const std::string& type,
                                                        const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTSuperLayerId slId(rawid);

  // Slayer specific parameter (size)
  vector<double> par = extractParameters(fv);

  edm::LogVerbatim("DTGeometryBuilder") << "(2) detId: " << rawid << " par[0]: " << par[0] << " par[1]: " << par[1]
                                        << " par[2]: " << par[2];

  // r-phi  dimension - different in different chambers
  // z      dimension - constant 126.8 cm
  // radial thickness - almost constant about 20 cm

  // Ok this is the s-layer position...
  RCPPlane surf(plane(fv, dtGeometryBuilder::getRecPlaneBounds(par.begin())));

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
                                              const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);

  // Layer specific parameter (size)
  std::vector<double> par = extractParameters(fv);
  // width -- r-phi  dimension - different in different chambers
  // length -- z      dimension - constant 126.8 cm
  // thickness -- radial thickness - almost constant about 20 cm

  RCPPlane surf(plane(fv, dtGeometryBuilder::getRecPlaneBounds(par.begin())));

  edm::LogVerbatim("DTGeometryBuilder") << "(3) detId: " << rawid << " par[0]: " << par[0] << " par[1]: " << par[1]
                                        << " par[2]: " << par[2];

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter = 0;
  int firstWire = fv.copyno();
  par = extractParameters(fv);
  float wireLength = convertMmToCm(par[1]);

  edm::LogVerbatim("DTGeometryBuilder") << "(4) detId: " << rawid
                                        << " wireLenght in ddd, wpar[1] in dd4hep: " << wireLength
                                        << " firstWire: " << firstWire;

  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling();  // next wire
  }
  //int lastWire=fv.copyno();
  DTTopology topology(firstWire, WCounter, wireLength);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);

  sl->add(layer);
  return layer;
}

vector<double> DTGeometryBuilderFromDDD::extractParameters(DDFilteredView& fv) const {
  vector<double> par;
  if (fv.logicalPart().solid().shape() != DDSolidShape::ddbox) {
    DDBooleanSolid bs(fv.logicalPart().solid());
    DDSolid A = bs.solidA();
    while (A.shape() != DDSolidShape::ddbox) {
      DDBooleanSolid bs(A);
      A = bs.solidA();
    }
    par = A.parameters();
  } else {
    par = fv.logicalPart().solid().parameters();
  }
  return par;
}

DTGeometryBuilderFromDDD::RCPPlane DTGeometryBuilderFromDDD::plane(const DDFilteredView& fv, Bounds* bounds) const {
  // extract the position
  const DDTranslation& trans(fv.translation());

  const Surface::PositionType posResult(
      float(convertMmToCm(trans.x())), float(convertMmToCm(trans.y())), float(convertMmToCm(trans.z())));
  LogTrace("DTGeometryBuilderFromDDD") << "DTGeometryBuilderFromDDD::plane  posResult: " << posResult;
  // now the rotation
  //     'active' and 'passive' rotations are inverse to each other
  const DDRotationMatrix& rotation = fv.rotation();  //REMOVED .Inverse();
  DD3Vector x, y, z;
  rotation.GetComponents(x, y, z);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "INVERSE rotation by its own operator: " << fv.rotation();
  edm::LogVerbatim("DTGeometry") << "INVERSE rotation manually: " << x.X() << ", " << x.Y() << ", " << x.Z()
                                 << std::endl
                                 << y.X() << ", " << y.Y() << ", " << y.Z() << std::endl
                                 << z.X() << ", " << z.Y() << ", " << z.Z();
#endif
  Surface::RotationType rotResult(float(x.X()),
                                  float(x.Y()),
                                  float(x.Z()),
                                  float(y.X()),
                                  float(y.Y()),
                                  float(y.Z()),
                                  float(z.X()),
                                  float(z.Y()),
                                  float(z.Z()));

  return RCPPlane(new Plane(posResult, rotResult, bounds));
}
