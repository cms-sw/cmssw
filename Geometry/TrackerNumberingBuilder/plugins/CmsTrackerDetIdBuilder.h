#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H
# define Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H

# include "FWCore/ParameterSet/interface/types.h"
# include <ostream>

class GeometricDet;

/**
 * Class to build a geographicalId.
 */

class CmsTrackerDetIdBuilder
{
public:
  CmsTrackerDetIdBuilder( unsigned int layerNumberPXB );
  GeometricDet* buildId( GeometricDet *det );  
protected:
  void iterate( GeometricDet const *det, int level, unsigned int ID );
  
private:
  // This is the map between detid and navtype to restore backward compatibility between 12* and 13* series
  std::map< std::string , uint32_t > m_mapNavTypeToDetId;
  unsigned int m_layerNumberPXB;
};

#endif
