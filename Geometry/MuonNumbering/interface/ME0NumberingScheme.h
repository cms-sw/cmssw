#ifndef Geometry_MuonNumbering_ME0NumberingScheme_h
#define Geometry_MuonNumbering_ME0NumberingScheme_h

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class MuonGeometryConstants;

class ME0NumberingScheme : public MuonNumberingScheme {
public:
  ME0NumberingScheme(const MuonGeometryConstants& muonConstants);

  ~ME0NumberingScheme() override{};

  int baseNumberToUnitNumber(const MuonBaseNumber&) const override;

private:
  void initMe(const MuonGeometryConstants& muonConstants);

  int theRegionLevel;
  int theSectorLevel;
  int theLayerLevel;
  int theRollLevel;
  int theNEtaPart;
};

#endif
