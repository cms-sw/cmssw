#ifndef Geometry_MTDNumberingBuilder_CmsMTDModuleBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDModuleBuilder_H

#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds Pixel Ladders
 */
class CmsMTDModuleBuilder : public CmsMTDLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricTimingDet*) override;
  void buildComponent(DDFilteredView& , GeometricTimingDet*, std::string) override;

};

#endif
