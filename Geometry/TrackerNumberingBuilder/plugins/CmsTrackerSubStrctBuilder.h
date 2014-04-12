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
  CmsTrackerSubStrctBuilder( unsigned int totalBlade );
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
  unsigned int m_totalBlade;
};

#endif
