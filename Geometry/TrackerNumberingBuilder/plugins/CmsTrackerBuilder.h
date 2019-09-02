#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Abstract Class to construct a Level in the hierarchy
 */
template <class T>
class CmsTrackerBuilder : public CmsTrackerLevelBuilder<T> {
public:
  CmsTrackerBuilder() {}

private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
