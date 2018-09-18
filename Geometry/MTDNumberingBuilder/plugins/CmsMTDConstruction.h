#ifndef Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#define Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
#include<string>
#include<vector>
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
/**
 * Adds GeometricTimingDets representing final modules to the previous level
 */
class CmsMTDConstruction : public CmsMTDLevelBuilder {
 public:
  void  buildComponent(DDFilteredView& , GeometricTimingDet*, std::string) override;
 private:

  void buildBTLModule(DDFilteredView& , GeometricTimingDet* , const std::string&);
  void buildETLModule(DDFilteredView& , GeometricTimingDet* , const std::string&);
};

#endif // Geometry_MTDNumberingBuilder_CmsMTDConstruction_H
