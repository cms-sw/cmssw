#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2DiskBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2DiskBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which contructs Phase2 Pixel Tracker/Discs.
 */
class CmsTrackerPixelPhase2DiskBuilder : public CmsTrackerLevelBuilder
{
  
private:
  void sortNS( DDFilteredView& , GeometricDet* ) override;
  void buildComponent( DDFilteredView& , GeometricDet*, std::string ) override;
  
};

#endif
