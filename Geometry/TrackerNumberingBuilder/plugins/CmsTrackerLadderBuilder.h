#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds Pixel Ladders
 */
template <class T>
class CmsTrackerLadderBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
