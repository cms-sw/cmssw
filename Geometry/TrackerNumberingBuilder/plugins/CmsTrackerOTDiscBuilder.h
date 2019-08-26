#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs Phase2 Outer Tracker/Discs.
 */
template <class T>
class CmsTrackerOTDiscBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
