#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerSubStrctBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Classes which abuilds all the tracker substructures
 */
template <class T>
class CmsTrackerSubStrctBuilder : public CmsTrackerLevelBuilder<T> {
public:
  CmsTrackerSubStrctBuilder() {}

private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
