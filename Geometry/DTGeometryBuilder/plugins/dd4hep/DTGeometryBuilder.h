#ifndef GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H
#define GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H

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
//         Modified by Sergio Lo Meo (sergio.lo.meo@cern.ch) Mon, 31 August 2020
//
//

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
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

namespace cms {

  class DDDetector;
  class DDFilteredView;

  class DTGeometryBuilder {
  public:
    DTGeometryBuilder() {}

    void build(DTGeometry&, const DDDetector*, const MuonGeometryConstants&, const dd4hep::SpecParRefs&);

  private:
    void buildGeometry(DDFilteredView&, DTGeometry&, const MuonGeometryConstants&) const;

    DTChamber* buildChamber(DDFilteredView&, const MuonGeometryConstants&) const;

    DTSuperLayer* buildSuperLayer(DDFilteredView&, DTChamber*, const MuonGeometryConstants&) const;

    DTLayer* buildLayer(DDFilteredView&, DTSuperLayer*, const MuonGeometryConstants&) const;

    using RCPPlane = ReferenceCountingPointer<Plane>;

    RCPPlane plane(const DDFilteredView&, Bounds* bounds) const;
  };
}  // namespace cms

#endif
