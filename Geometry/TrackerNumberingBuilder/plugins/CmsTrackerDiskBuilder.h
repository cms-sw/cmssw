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
public:
  CmsTrackerDiskBuilder( unsigned int totalBlade );
  
private:
  virtual void sortNS( DDFilteredView& , GeometricDet* );
  virtual void buildComponent( DDFilteredView& , GeometricDet*, std::string );
  
  void PhiPosNegSplit_innerOuter( GeometricDet::ConstGeometricDetContainer::iterator begin,
				  GeometricDet::ConstGeometricDetContainer::iterator end );
  unsigned int m_totalBlade;
};

#endif
