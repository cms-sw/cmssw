#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/ModifiedSurfaceGenerator.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

GeomDet::~GeomDet() { delete theAlignmentPositionError; }

void GeomDet::move(const GlobalVector& displacement) {
  //
  // Should recreate the surface like the set* methods ?
  //
  thePlane->move(displacement);
}

void GeomDet::rotate(const Surface::RotationType& rotation) {
  //
  // Should recreate the surface like the set* methods ?
  //
  thePlane->rotate(rotation);
}

void GeomDet::setPosition(const Surface::PositionType& position, const Surface::RotationType& rotation) {
  thePlane = ModifiedSurfaceGenerator<Plane>(thePlane).atNewPosition(position, rotation);
}

bool GeomDet::setAlignmentPositionError(const AlignmentPositionError& ape) {
  if (!theAlignmentPositionError) {
    if (ape.valid())
      theAlignmentPositionError = new AlignmentPositionError(ape);
  } else
    *theAlignmentPositionError = ape;
  return ape.valid();
}

#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/GeomDetType.h"
#include "FWCore/Utilities/interface/Exception.h"

GeomDet::SubDetector GeomDet::subDetector() const { return type().subDetector(); }

void GeomDet::setSurfaceDeformation(const SurfaceDeformation* /*deformation*/) {
  throw cms::Exception("Geometry") << "setting SurfaceDeformation not implemented for DetId "
                                   << geographicalId().rawId() << " det=" << geographicalId().det()
                                   << " subdetId=" << geographicalId().subdetId();
}

#include "Geometry/CommonTopologies/interface/Topology.h"

namespace {
  struct DummyTopology final : public Topology {
    LocalPoint localPosition(const MeasurementPoint&) const override { return LocalPoint(); }
    LocalError localError(const MeasurementPoint&, const MeasurementError&) const override { return LocalError(); }
    MeasurementPoint measurementPosition(const LocalPoint&) const override { return MeasurementPoint(); }
    MeasurementError measurementError(const LocalPoint&, const LocalError&) const override {
      return MeasurementError();
    }
    int channel(const LocalPoint& p) const override { return -1; }
  };
  const DummyTopology dummyTopology{};

  struct DummyGeomDetType final : public GeomDetType {
    DummyGeomDetType() : GeomDetType("", GeomDetEnumerators::invalidDet) {}
    const Topology& topology() const override { return dummyTopology; }
  };
  const DummyGeomDetType dummyGeomDetType{};
}  // namespace

const Topology& GeomDet::topology() const { return dummyTopology; }

const GeomDetType& GeomDet::type() const { return dummyGeomDetType; }
