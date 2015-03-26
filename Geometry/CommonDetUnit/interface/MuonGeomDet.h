#ifndef CommonDet_MuonGeomDet_H
#define CommonDet_MuonGeomDet_H


#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"

class MuonGeomDet : public GeomDet {
protected :
  explicit MuonGeomDet(Plane * plane) : GeomDet(plane), theLocalAlignmentError(InvalidError()){}
  explicit MuonGeomDet(const ReferenceCountingPointer<Plane>& plane) : GeomDet(plane), theLocalAlignmentError(InvalidError()){}

public:
  /// Return local alligment error
  LocalErrorExtended const & localAlignmentError() const { return theLocalAlignmentError;}

private:

  LocalErrorExtended  theLocalAlignmentError;

private:
  bool setAlignmentPositionError (const AlignmentPositionError& ape) GCC11_FINAL;

};
#endif
