#ifndef Geometry_MTDNumberingBuilder_CmsMTDETLRingBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDETLRingBuilder_H

#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs MTD ETL Rings. 
 */
class CmsMTDETLRingBuilder : public CmsMTDLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricTimingDet*) override;
  void buildComponent(DDFilteredView& , GeometricTimingDet*, std::string) override;

};

#endif
