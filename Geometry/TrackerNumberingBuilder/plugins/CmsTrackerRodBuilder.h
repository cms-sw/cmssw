#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TOB rods
 */
template <class T>
class CmsTrackerRodBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
