#ifndef Geometry_MTDNumberingBuilder_DDDCmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_DDDCmsMTDConstruction_H

#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
#include <vector>
#include <memory>

class GeometricTimingDet;
class DDCompactView;

namespace cms {
  class DDCompactView;
}

/**
 * High level class to build a tracker. It will only build subdets,
 * then call subdet builders
 */

class DDDCmsMTDConstruction {
public:
  DDDCmsMTDConstruction() = delete;
  static std::unique_ptr<GeometricTimingDet> construct(const DDCompactView& cpv);
  static std::unique_ptr<GeometricTimingDet> construct(const cms::DDCompactView& cpv);
private:
  static constexpr size_t kNLayerPreTDR = 3;
  static constexpr size_t kNLayerTDR = 5;
};

#endif
