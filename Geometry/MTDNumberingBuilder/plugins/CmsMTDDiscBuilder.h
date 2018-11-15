#ifndef Geometry_MTDNumberingBuilder_CmsMTDDiscBuilder_H
# define Geometry_MTDNumberingBuilder_CmsMTDDiscBuilder_H

# include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which contructs Phase2 Outer Tracker/Discs.
 */
class CmsMTDDiscBuilder : public CmsMTDLevelBuilder
{
  
private:
  void sortNS( DDFilteredView& , GeometricTimingDet* ) override;
  void buildComponent( DDFilteredView& , GeometricTimingDet*, std::string ) override;
  
};

#endif
