#ifndef GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H
#define GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/MuonNumbering/interface/DD4hep_DTNumberingScheme.h"

#include <DD4hep/SpecParRegistry.h>

namespace dd4hep {
  class Detector;
}

class DTGeometry;
class DTChamber;
class DTSuperLayer;
class DTLayer;

namespace cms {

  class DDDetector;
  class DDFilteredView;
  class MuonNumbering;

  class DTGeometryBuilder {
  public:
    DTGeometryBuilder() {}

    void build(DTGeometry&, const DDDetector*, const MuonNumbering&, const dd4hep::SpecParRefs&);

  private:
    void buildGeometry(DDFilteredView&, DTGeometry&, const MuonNumbering&) const;

    /// create the chamber
    DTChamber* buildChamber(DDFilteredView&, const MuonNumbering&) const;

    /// create the SL
    DTSuperLayer* buildSuperLayer(DDFilteredView&, DTChamber*, const MuonNumbering&) const;

    /// create the layer
    DTLayer* buildLayer(DDFilteredView&, DTSuperLayer*, const MuonNumbering&) const;

    using RCPPlane = ReferenceCountingPointer<Plane>;

    RCPPlane plane(const DDFilteredView&, Bounds* bounds) const;

    std::unique_ptr<cms::DTNumberingScheme> dtnum_ = nullptr;
  };
}  // namespace cms

#endif
