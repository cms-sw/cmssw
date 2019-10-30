// -*- C++ -*-
//
// Package:    DetectorDescription/DTGeometryBuilder
// Class:      DTGeometryBuilder
//
/**\class DTGeometryBuilder

 Description: DT Geometry builder from DD4hep

 Implementation:
     DT Geometry Builder iterates over a Detector Tree and
     retrvieves DT chambers, super layers, layers and wires.
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 16 Jan 2019 10:19:37 GMT
//
//
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/MuonNumbering/interface/DD4hep_DTNumberingScheme.h"
#include "DTGeometryBuilder.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>
#include <string_view>

using namespace edm;
using namespace std;
using namespace cms;

void DTGeometryBuilder::buildGeometry(DDFilteredView& fview, DTGeometry& geom, const MuonNumbering& num) const {
  bool doChamber = fview.firstChild();

  while (doChamber) {
    DTChamber* chamber = buildChamber(fview, num);

    // Loop on SLs
    bool doSL = fview.nextSibling();
    while (doSL) {
      DTSuperLayer* sl = buildSuperLayer(fview, chamber, num);

      // Loop on Layers
      fview.down();
      bool doLayers = fview.sibling();
      while (doLayers) {
        DTLayer* l = buildLayer(fview, sl, num);
        geom.add(l);

        doLayers = fview.sibling();  // go to next Layer
      }
      // Done with layers

      geom.add(sl);
      doSL = fview.nextSibling();  // go to next SL
    }
    geom.add(chamber);

    fview.parent();                  // stop iterating current branch
    doChamber = fview.firstChild();  // go to next chamber
  }
}

DTGeometryBuilder::RCPPlane DTGeometryBuilder::plane(const DDFilteredView& fview, Bounds* bounds) const {
  const Double_t* tr = fview.trans();
  const Double_t* rot = fview.rot();

  return RCPPlane(
      new Plane(Surface::PositionType(tr[0], tr[1], tr[2]),
                Surface::RotationType(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]),
                bounds));
}

DTChamber* DTGeometryBuilder::buildChamber(DDFilteredView& fview, const MuonNumbering& muonConstants) const {
  int rawid = dtnum_->getDetId(muonConstants.geoHistoryToBaseNumber(fview.history()));
  DTChamberId detId(rawid);
  auto const& par = fview.parameters();
  // par[0] r-phi  dimension - different in different chambers
  // par[1] z      dimension - constant 125.55 cm
  // par[2] radial thickness - almost constant about 18 cm

  RCPPlane surf(plane(fview, new RectangularPlaneBounds(par[0], par[1], par[2])));

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer* DTGeometryBuilder::buildSuperLayer(DDFilteredView& fview,
                                                 DTChamber* chamber,
                                                 const MuonNumbering& muonConstants) const {
  int rawid = dtnum_->getDetId(muonConstants.geoHistoryToBaseNumber(fview.history()));
  DTSuperLayerId slId(rawid);

  auto const& par = fview.parameters();
  // par[0] r-phi  dimension - changes in different chambers
  // par[1] z      dimension - constant 126.8 cm
  // par[2] radial thickness - almost constant about 20 cm

  // Ok this is the slayer position...
  RCPPlane surf(plane(fview, new RectangularPlaneBounds(par[0], par[1], par[2])));

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  // add to the chamber
  chamber->add(slayer);

  return slayer;
}

DTLayer* DTGeometryBuilder::buildLayer(DDFilteredView& fview,
                                       DTSuperLayer* sl,
                                       const MuonNumbering& muonConstants) const {
  int rawid = dtnum_->getDetId(muonConstants.geoHistoryToBaseNumber(fview.history()));
  DTLayerId layId(rawid);

  auto const& par = fview.parameters();
  // Layer specific parameter (size)
  // par[0] r-phi  dimension - changes in different chambers
  // par[1] z      dimension - constant 126.8 cm
  // par[2] radial thickness - almost constant about 20 cm
  RCPPlane surf(plane(fview, new RectangularPlaneBounds(par[0], par[1], par[2])));

  // Loop on wires
  fview.down();
  bool doWire = fview.sibling();
  int firstWire = fview.volume()->GetNumber();  // copy no
  auto const& wpar = fview.parameters();
  float wireLength = wpar[1];

  int WCounter = 0;
  while (doWire) {
    doWire = fview.checkChild();
    WCounter++;
  }
  fview.up();

  DTTopology topology(firstWire, WCounter, wireLength);

  DTLayerType layerType;

  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);

  sl->add(layer);
  return layer;
}

void DTGeometryBuilder::build(DTGeometry& geom,
                              const DDDetector* det,
                              const MuonNumbering& num,
                              const DDSpecParRefs& refs) {
  Volume top = det->worldVolume();
  DDFilteredView fview(det, top);
  fview.mergedSpecifics(refs);
  dtnum_ = make_unique<DTNumberingScheme>(num.values());
  buildGeometry(fview, geom, num);
}
