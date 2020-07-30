#ifndef Geometry_MuonNumbering_GEMNumberingScheme_h
#define Geometry_MuonNumbering_GEMNumberingScheme_h

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class MuonGeometryConstants;

class GEMNumberingScheme : public MuonNumberingScheme {
public:
  GEMNumberingScheme(const MuonGeometryConstants& muonConstants);

  ~GEMNumberingScheme() override{};

  int baseNumberToUnitNumber(const MuonBaseNumber&) const override;

private:
  void initMe(const MuonGeometryConstants& muonConstants);

  int theRegionLevel;
  int theStationLevel;
  int theRingLevel;
  int theSectorLevel;
  int theRollLevel;
};

#endif
