#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPanelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPanelBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward Panels. 
 */
template <class T>
class CmsTrackerPanelBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
