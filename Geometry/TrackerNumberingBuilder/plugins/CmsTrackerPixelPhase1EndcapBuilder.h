#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which builds the pixel phase 1 endcap
 */
class CmsTrackerPixelPhase1EndcapBuilder : public CmsTrackerLevelBuilder
{
public:
  CmsTrackerPixelPhase1EndcapBuilder();
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
};

#endif
