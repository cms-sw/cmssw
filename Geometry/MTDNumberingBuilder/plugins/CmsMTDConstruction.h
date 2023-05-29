#ifndef Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#include <string>
#include <vector>
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

/**
 * Adds GeometricTimingDets representing final modules to the previous level
 */
template <class FilteredView>
class CmsMTDConstruction {
public:
  CmsMTDConstruction();
  ~CmsMTDConstruction() = default;

  static bool mtdOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool mtdOrderRR(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool mtdOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool btlOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b);
  static bool btlOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b);

  void buildBTLModule(FilteredView&, GeometricTimingDet*);
  void buildETLModule(FilteredView&, GeometricTimingDet*);

  GeometricTimingDet* buildSubdet(FilteredView&);
  GeometricTimingDet* buildLayer(FilteredView&);

  void baseNumberFromHistory(const DDGeoHistory& gh);

  bool isBTLV2(FilteredView&);

  bool isETLtdr(FilteredView&);

protected:
  CmsMTDStringToEnum theCmsMTDStringToEnum;

  BTLNumberingScheme btlScheme_;
  ETLNumberingScheme etlScheme_;
  MTDBaseNumber baseNumber_;
};

#endif  // Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
