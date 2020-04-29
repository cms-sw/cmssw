#ifndef MuonNumbering_MuonDDDParametersBuild_h
#define MuonNumbering_MuonDDDParametersBuild_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class MuonDDDParameters;

class MuonDDDParametersBuild {

public:
  MuonDDDParametersBuild() {}

  bool build(const DDCompactView* cpv, MuonDDDParameters& php);
  bool build(const cms::DDCompactView* cpv, MuonDDDParameters& php);
};

#endif
