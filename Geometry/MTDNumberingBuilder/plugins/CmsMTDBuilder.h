#ifndef Geometry_MTDNumberingBuilder_CmsMTDBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDBuilder_H

# include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Abstract Class to construct a Level in the hierarchy
 */
class CmsMTDBuilder : public CmsMTDLevelBuilder
{
public:
  CmsMTDBuilder();

private:

  void sortNS( DDFilteredView& , GeometricTimingDet* ) override;
  void buildComponent( DDFilteredView& , GeometricTimingDet*, std::string ) override;
};

#endif
