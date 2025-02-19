#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerDetIdBuilder_H

#include "FWCore/ParameterSet/interface/types.h"
#include <ostream>

class GeometricDet;

/**
 * Class to build a geographicalId.
 */

class CmsTrackerDetIdBuilder {
 public:
  CmsTrackerDetIdBuilder();
  GeometricDet* buildId(GeometricDet*);  
 protected:
  void iterate(GeometricDet const *,int,unsigned int );
  
private:
  // This is the map between detid and navtype to restore backward compatibility between 12* and 13* series
  std::map< std::string , uint32_t > mapNavTypeToDetId;
  //
};

#endif
