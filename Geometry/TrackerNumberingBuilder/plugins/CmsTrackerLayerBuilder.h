#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs TIB/TOB layers
 */
template <class T>
class CmsTrackerLayerBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
