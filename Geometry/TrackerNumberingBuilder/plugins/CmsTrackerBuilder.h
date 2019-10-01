#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Abstract Class to construct a Level in the hierarchy
 */
template <class FilteredView>
class CmsTrackerBuilder : public CmsTrackerLevelBuilder<FilteredView> {
public:
  CmsTrackerBuilder() {}

private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
