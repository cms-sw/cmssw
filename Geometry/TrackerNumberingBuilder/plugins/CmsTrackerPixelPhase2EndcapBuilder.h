#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which builds the pixel phase 1 endcap
 */
class CmsTrackerPixelPhase2EndcapBuilder : public CmsTrackerLevelBuilder
{
public:
  CmsTrackerPixelPhase2EndcapBuilder();
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
};

#endif
