// -*- C++ -*-
//
// Package:    Geometry/DTGeometryBuilder
// Class:      DTGeometryBuilderFromDD4Hep
//
/**\class DTGeometryBuilderFromDD4Hep

 Description: DT Geometry builder from DD4hep

 Implementation:
     DT Geometry Builder iterates over a Detector Tree and
     retrvieves DT chambers, super layers, layers and wires.
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 16 Jan 2019 10:19:37 GMT
//         Modified by Sergio Lo Meo (sergio.lo.meo@cern.ch) Mon, 31 August 2020
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
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DTGeometryBuilderFromDD4Hep.h"
#include "DD4hep/Detector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <string>
#include <string_view>

using namespace edm;
using namespace std;

void DTGeometryBuilderFromDD4Hep::buildGeometry(cms::DDFilteredView& fview,
                                                DTGeometry& geom,
                                                const MuonGeometryConstants& num) const {
  edm::LogVerbatim("DTGeometryBuilder") << "(0) DTGeometryBuilder - DD4Hep ";

  bool doChamber = fview.firstChild();

  while (doChamber) {
    DTChamber* chamber = buildChamber(fview, num);

    bool doSL = fview.nextSibling();
    while (doSL) {
      DTSuperLayer* sl = buildSuperLayer(fview, chamber, num);

      fview.down();
      bool doLayers = fview.sibling();
      while (doLayers) {
        DTLayer* l = buildLayer(fview, sl, num);
        geom.add(l);

        doLayers = fview.sibling();
      }

      geom.add(sl);
      doSL = fview.nextSibling();
    }
    geom.add(chamber);

    fview.parent();
    doChamber = fview.firstChild();
  }
}

DTGeometryBuilderFromDD4Hep::RCPPlane DTGeometryBuilderFromDD4Hep::plane(const cms::DDFilteredView& fview,
                                                                         Bounds* bounds) const {
  const Double_t* tr = fview.trans();
  const Double_t* rot = fview.rot();

  return RCPPlane(
      new Plane(Surface::PositionType(tr[0] / dd4hep::cm, tr[1] / dd4hep::cm, tr[2] / dd4hep::cm),
                Surface::RotationType(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]),
                bounds));
}

DTChamber* DTGeometryBuilderFromDD4Hep::buildChamber(cms::DDFilteredView& fview,
                                                     const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fview.history()));

  DTChamberId detId(rawid);
  auto const& par = fview.parameters();

  RCPPlane surf(
      plane(fview, new RectangularPlaneBounds(par[0] / dd4hep::cm, par[1] / dd4hep::cm, par[2] / dd4hep::cm)));

  edm::LogVerbatim("DTGeometryBuilder") << "(1) detId: " << rawid << " par[0]: " << par[0] / dd4hep::cm
                                        << " par[1]: " << par[1] / dd4hep::cm << " par[2]: " << par[2] / dd4hep::cm;

  DTChamber* chamber = new DTChamber(detId, surf);

  return chamber;
}

DTSuperLayer* DTGeometryBuilderFromDD4Hep::buildSuperLayer(cms::DDFilteredView& fview,
                                                           DTChamber* chamber,
                                                           const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fview.history()));

  DTSuperLayerId slId(rawid);

  auto const& par = fview.parameters();

  RCPPlane surf(
      plane(fview, new RectangularPlaneBounds(par[0] / dd4hep::cm, par[1] / dd4hep::cm, par[2] / dd4hep::cm)));

  edm::LogVerbatim("DTGeometryBuilder") << "(2) detId: " << rawid << " par[0]: " << par[0] / dd4hep::cm
                                        << " par[1]: " << par[1] / dd4hep::cm << " par[2]: " << par[2] / dd4hep::cm;

  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);

  chamber->add(slayer);

  return slayer;
}

DTLayer* DTGeometryBuilderFromDD4Hep::buildLayer(cms::DDFilteredView& fview,
                                                 DTSuperLayer* sl,
                                                 const MuonGeometryConstants& muonConstants) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fview.history()));

  DTLayerId layId(rawid);

  auto const& par = fview.parameters();

  RCPPlane surf(
      plane(fview, new RectangularPlaneBounds(par[0] / dd4hep::cm, par[1] / dd4hep::cm, par[2] / dd4hep::cm)));

  edm::LogVerbatim("DTGeometryBuilder") << "(3) detId: " << rawid << " par[0]: " << par[0] / dd4hep::cm
                                        << " par[1]: " << par[1] / dd4hep::cm << " par[2]: " << par[2] / dd4hep::cm;

  fview.down();
  bool doWire = fview.sibling();
  int firstWire = fview.volume()->GetNumber();
  auto const& wpar = fview.parameters();
  float wireLength = wpar[1] / dd4hep::cm;

  edm::LogVerbatim("DTGeometryBuilder") << "(4) detId: " << rawid << " wpar[1]: " << wpar[1] / dd4hep::cm
                                        << " firstWire: " << firstWire;

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

void DTGeometryBuilderFromDD4Hep::build(DTGeometry& geom,
                                        const cms::DDDetector* det,
                                        const MuonGeometryConstants& num,
                                        const dd4hep::SpecParRefs& refs) {
  cms::Volume top = det->worldVolume();
  cms::DDFilteredView fview(det, top);
  fview.mergedSpecifics(refs);
  buildGeometry(fview, geom, num);
}
