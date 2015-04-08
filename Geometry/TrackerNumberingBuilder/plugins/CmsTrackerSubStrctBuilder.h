#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Classes which abuilds all the tracker substructures
 */
class CmsTrackerSubStrctBuilder : public CmsTrackerLevelBuilder
{
public:
  CmsTrackerSubStrctBuilder();
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
};

#endif
