#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDebugNavigator_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerDebugNavigator_H

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"

class GeometricDet;
/**
 * This class travel recursively a GeometricDet and dumps the information about type
 */
class CmsTrackerDebugNavigator {
 public:
void  dump(const GeometricDet*);
 private:
 void iterate(const GeometricDet*,int);
 int numinstances[30];
 CmsTrackerStringToEnum _CmsTrackerStringToEnum;
};

#endif
