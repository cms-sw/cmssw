#ifndef Geometry_MTDNumberingBuilder_CmsMTDSubStrctBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDSubStrctBuilder_H

# include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Classes which abuilds all the tracker substructures
 */
class CmsMTDSubStrctBuilder : public CmsMTDLevelBuilder
{
public:
  CmsMTDSubStrctBuilder();
  
private:
  void sortNS( DDFilteredView& , GeometricTimingDet* ) override;
  void buildComponent( DDFilteredView& , GeometricTimingDet*, std::string ) override;
};

#endif
