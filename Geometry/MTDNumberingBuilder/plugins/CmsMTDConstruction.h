#ifndef Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#include <string>
#include <vector>
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

/**
 * Adds GeometricTimingDets representing final modules to the previous level
 */
class CmsMTDConstruction {
public:
  CmsMTDConstruction();
  ~CmsMTDConstruction() = default;

  static bool mtdOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool mtdOrderRR(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool mtdOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b);

  void buildBTLModule(DDFilteredView&, GeometricTimingDet*, const std::string&);
  void buildETLModule(DDFilteredView&, GeometricTimingDet*, const std::string&);

  GeometricTimingDet* buildSubdet(DDFilteredView&, GeometricTimingDet*, const std::string&);
  GeometricTimingDet* buildLayer(DDFilteredView&, GeometricTimingDet*, const std::string&);

  void baseNumberFromHistory(const DDGeoHistory& gh);

  bool isETLtdr(DDFilteredView&);

protected:
  CmsMTDStringToEnum theCmsMTDStringToEnum;

  ETLNumberingScheme etlScheme_;
  MTDBaseNumber baseNumber_;
};

#endif  // Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
