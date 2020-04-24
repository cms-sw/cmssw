#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H

# include "FWCore/ParameterSet/interface/types.h"
# include <ostream>
#include <vector>
#include <array>

class GeometricDet;

/**
 * Class to build a geographicalId.
 */

class CmsTrackerDetIdBuilder
{
public:
  CmsTrackerDetIdBuilder(std::vector<int> detidShifts );
  GeometricDet* buildId( GeometricDet *det );  
protected:
  void iterate( GeometricDet *det, int level, unsigned int ID );
  
private:

  static const unsigned int nSubDet=6;
  static const int maxLevels=6;

  // This is the map between detid and navtype to restore backward compatibility between 12* and 13* series
  std::map< std::string , uint32_t > m_mapNavTypeToDetId;
  std::array<int,nSubDet*maxLevels> m_detidshifts; 
};

#endif
