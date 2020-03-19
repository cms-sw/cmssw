#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs TIB/TOB layers
 */
template <class FilteredView>
class CmsTrackerOTLayerBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
