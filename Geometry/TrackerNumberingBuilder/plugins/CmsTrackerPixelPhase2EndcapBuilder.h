#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which builds the pixel phase 2 endcap
 */
class CmsTrackerPixelPhase2EndcapBuilder : public CmsTrackerLevelBuilder
{
public:
  CmsTrackerPixelPhase2EndcapBuilder();
  
private:
  void sortNS( DDFilteredView& , GeometricDet* ) override;
  void buildComponent( DDFilteredView& , GeometricDet*, std::string ) override;
};

#endif
