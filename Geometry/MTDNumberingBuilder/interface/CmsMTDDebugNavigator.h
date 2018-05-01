#ifndef Geometry_MTDNumberingBuilder_CmsMTDDebugNavigator_H
#define Geometry_MTDNumberingBuilder_CmsMTDDebugNavigator_H

#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDetExtra.h"

#include <vector>

class GeometricTimingDet;
/**
 * This class travel recursively a GeometricTimingDet and dumps the information about type
 */
class CmsMTDDebugNavigator {
 public:
  CmsMTDDebugNavigator (const std::vector<GeometricTimingDetExtra> & );
  void  dump(const GeometricTimingDet&, const std::vector<GeometricTimingDetExtra> & );
 private:
  void iterate(const GeometricTimingDet&,int, const std::vector<GeometricTimingDetExtra> & );
  int numinstances[30];
  CmsMTDStringToEnum _CmsMTDStringToEnum;
  std::map<uint32_t, const GeometricTimingDetExtra*> _helperMap; 
};

#endif
