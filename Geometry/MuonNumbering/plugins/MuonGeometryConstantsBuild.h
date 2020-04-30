#ifndef MuonNumbering_MuonGeometryConstantsBuild_h
#define MuonNumbering_MuonGeometryConstantsBuild_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class MuonGeometryConstants;

class MuonGeometryConstantsBuild {
public:
  MuonGeometryConstantsBuild() {}

  bool build(const DDCompactView* cpv, MuonGeometryConstants& php);
  bool build(const cms::DDCompactView* cpv, MuonGeometryConstants& php);
};

#endif
