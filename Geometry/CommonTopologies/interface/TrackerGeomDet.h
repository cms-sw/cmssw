#ifndef CommonDet_TrackerGeomDet_H
#define CommonDet_TrackerGeomDet_H

#include "Geometry/CommonTopologies/interface/GeomDet.h"

class TrackerGeomDet : public GeomDet {
protected:
  explicit TrackerGeomDet(Plane* plane) : GeomDet(plane), theLocalAlignmentError(InvalidError()) {}
  explicit TrackerGeomDet(const ReferenceCountingPointer<Plane>& plane)
      : GeomDet(plane), theLocalAlignmentError(InvalidError()) {}

public:
  /// Return local alligment error
  LocalError const& localAlignmentError() const { return theLocalAlignmentError; }

private:
  LocalError theLocalAlignmentError;

private:
  bool setAlignmentPositionError(const AlignmentPositionError& ape) final;
};
#endif
