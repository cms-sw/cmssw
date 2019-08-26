#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TIB strings
 */
template <class T>
class CmsTrackerStringBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
