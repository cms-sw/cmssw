#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs TIB/TOB layers
 */
template <class FilteredView>
class CmsTrackerLayerBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
