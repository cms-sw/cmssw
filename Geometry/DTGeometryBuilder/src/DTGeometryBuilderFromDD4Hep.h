#ifndef DTGeometryBuilder_DTGeometryBuilderFromDD4Hep_h
#define DTGeometryBuilder_DTGeometryBuilderFromDD4Hep_h
// -*- C++ -*-
//
// Package:    Gemetry/DTGeometryBuilder
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

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"

#include <DD4hep/SpecParRegistry.h>

namespace dd4hep {
  class Detector;
}

class DTGeometry;
class DTChamber;
class DTSuperLayer;
class DTLayer;
class MuonGeometryConstants;
class MuonGeometryNumbering;

class DTGeometryBuilderFromDD4Hep {
public:
  DTGeometryBuilderFromDD4Hep() {}

  void build(DTGeometry&, const cms::DDDetector*, const MuonGeometryConstants&, const dd4hep::SpecParRefs&);

private:
  void buildGeometry(cms::DDFilteredView&, DTGeometry&, const MuonGeometryConstants&) const;

  DTChamber* buildChamber(cms::DDFilteredView&, const MuonGeometryConstants&) const;

  DTSuperLayer* buildSuperLayer(cms::DDFilteredView&, DTChamber*, const MuonGeometryConstants&) const;

  DTLayer* buildLayer(cms::DDFilteredView&, DTSuperLayer*, const MuonGeometryConstants&) const;

  using RCPPlane = ReferenceCountingPointer<Plane>;

  RCPPlane plane(const cms::DDFilteredView&, Bounds* bounds) const;
};

#endif
