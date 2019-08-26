#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerWheelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerWheelBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds TEC wheels
 */
template <class T>
class CmsTrackerWheelBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
