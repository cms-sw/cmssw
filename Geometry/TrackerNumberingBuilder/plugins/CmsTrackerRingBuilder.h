#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerRingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerRingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TID/TEC rings
 */
template <class T>
class CmsTrackerRingBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
