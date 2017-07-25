#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which contructs PixelForward/Disk.
 */
class CmsTrackerDiskBuilder : public CmsTrackerLevelBuilder
{
  
private:
  void sortNS( DDFilteredView& , GeometricDet* ) override;
  void buildComponent( DDFilteredView& , GeometricDet*, std::string ) override;

};

#endif
