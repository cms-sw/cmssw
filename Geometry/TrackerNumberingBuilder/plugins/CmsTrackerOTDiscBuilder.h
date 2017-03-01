#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Class which contructs Phase2 Outer Tracker/Discs.
 */
class CmsTrackerOTDiscBuilder : public CmsTrackerLevelBuilder
{
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
  
};

#endif
