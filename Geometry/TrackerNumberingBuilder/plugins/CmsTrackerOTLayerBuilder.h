#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs TIB/TOB layers
 */
template <class T>
class CmsTrackerOTLayerBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
