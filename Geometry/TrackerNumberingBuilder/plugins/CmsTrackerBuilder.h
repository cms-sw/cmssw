#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H

# include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
# include "FWCore/ParameterSet/interface/types.h"
# include <string>

/**
 * Abstract Class to construct a Level in the hierarchy
 */
class CmsTrackerBuilder : public CmsTrackerLevelBuilder
{
public:
  CmsTrackerBuilder( unsigned int totalBlade );

private:
  unsigned int m_totalBlade;

  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
};

#endif
