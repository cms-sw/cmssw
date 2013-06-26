#ifndef MuonNumbering_GEMNumberingScheme_h
#define MuonNumbering_GEMNumberingScheme_h

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class DDCompactView;
class MuonDDDConstants;

class GEMNumberingScheme : public MuonNumberingScheme {

public:

  GEMNumberingScheme( const DDCompactView& cpv );
  GEMNumberingScheme( const MuonDDDConstants& muonConstants );

  virtual ~GEMNumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber);
  
private:
  void initMe ( const MuonDDDConstants& muonConstants );

  int theRegionLevel;
  int theStationLevel;
  int theRingLevel;
  int theSectorLevel;
  int theRollLevel;

};

#endif
