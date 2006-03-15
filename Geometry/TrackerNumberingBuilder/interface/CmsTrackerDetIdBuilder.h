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
  void iterate(GeometricDet*,int,unsigned int );
  
};

#endif
