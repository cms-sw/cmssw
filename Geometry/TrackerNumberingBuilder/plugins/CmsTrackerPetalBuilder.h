#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TEC petals
 */
template <class T>
class CmsTrackerPetalBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
