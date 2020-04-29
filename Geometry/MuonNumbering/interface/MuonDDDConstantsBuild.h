#ifndef MuonNumbering_MuonDDDConstantsBuild_h
#define MuonNumbering_MuonDDDConstantsBuild_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class MuonDDDParameters;

class MuonDDDConstantsBuild {
public:
  MuonDDDConstantsBuild() {}

  bool build(const DDCompactView* cpv, MuonDDDParameters& php);
  bool build(const cms::DDCompactView* cpv, MuonDDDParameters& php);
};

#endif
