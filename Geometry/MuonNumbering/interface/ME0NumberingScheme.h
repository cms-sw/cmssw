#ifndef Geometry_MuonNumbering_ME0NumberingScheme_h
#define Geometry_MuonNumbering_ME0NumberingScheme_h

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class DDCompactView;
class MuonDDDConstants;

class ME0NumberingScheme : public MuonNumberingScheme {

public:

  ME0NumberingScheme( const DDCompactView& cpv );
  ME0NumberingScheme( const MuonDDDConstants& muonConstants );
  
  virtual ~ME0NumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber&);
  
private:
  void initMe ( const MuonDDDConstants& muonConstants );

  int theRegionLevel;
  int theSectorLevel;
  int theLayerLevel;
  int theRollLevel;

};

#endif
