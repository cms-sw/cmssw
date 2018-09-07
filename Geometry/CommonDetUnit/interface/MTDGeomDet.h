#ifndef CommonDet_MTDGeomDet_H
#define CommonDet_MTDGeomDet_H


#include "Geometry/CommonDetUnit/interface/GeomDet.h"

class MTDGeomDet : public GeomDet {
protected :
  explicit MTDGeomDet(Plane * plane) : GeomDet(plane), theLocalAlignmentError(InvalidError()){}
  explicit MTDGeomDet(const ReferenceCountingPointer<Plane>& plane) : GeomDet(plane), theLocalAlignmentError(InvalidError()){}

public:
  /// Return local alligment error
  LocalError const & localAlignmentError() const { return theLocalAlignmentError;}

private:
  LocalError  theLocalAlignmentError;
  bool setAlignmentPositionError (const AlignmentPositionError& ape) final;

};
#endif
