#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Classes which abuilds all the tracker substructures
 */
template <class FilteredView>
class CmsTrackerSubStrctBuilder : public CmsTrackerLevelBuilder<FilteredView> {
public:
  CmsTrackerSubStrctBuilder() {}

private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
