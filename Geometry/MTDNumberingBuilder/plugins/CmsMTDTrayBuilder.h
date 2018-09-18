#ifndef Geometry_MTDNumberingBuilder_CmsMTDTrayBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDTrayBuilder_H

#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs BTL trays
 */
class CmsMTDTrayBuilder : public CmsMTDLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricTimingDet*) override;
  void buildComponent(DDFilteredView& , GeometricTimingDet*, std::string) override;

};

#endif
