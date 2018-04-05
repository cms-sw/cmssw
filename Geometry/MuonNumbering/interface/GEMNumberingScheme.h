#ifndef Geometry_MuonNumbering_GEMNumberingScheme_h
#define Geometry_MuonNumbering_GEMNumberingScheme_h

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class DDCompactView;
class MuonDDDConstants;

class GEMNumberingScheme : public MuonNumberingScheme {

public:

  GEMNumberingScheme( const DDCompactView& cpv );
  GEMNumberingScheme( const MuonDDDConstants& muonConstants );

  ~GEMNumberingScheme() override{};
  
  int baseNumberToUnitNumber(const MuonBaseNumber&) override;
  
private:
  void initMe ( const MuonDDDConstants& muonConstants );

  int theRegionLevel;
  int theStationLevel;
  int theRingLevel;
  int theSectorLevel;
  int theRollLevel;

};

#endif
