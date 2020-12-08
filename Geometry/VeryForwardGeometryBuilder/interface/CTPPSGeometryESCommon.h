#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

namespace CTPPSGeometryESCommon {
  std::unique_ptr<DetGeomDesc> applyAlignments(const DetGeomDesc&, const CTPPSRPAlignmentCorrectionsData*);
};