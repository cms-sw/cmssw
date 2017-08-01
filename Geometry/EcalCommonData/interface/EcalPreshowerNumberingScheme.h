///////////////////////////////////////////////////////////////////////////////
// File: EcalPreshowerNumberingScheme.h
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalPreshowerNumberingScheme_h
#define EcalPreshowerNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class EcalPreshowerNumberingScheme : public EcalNumberingScheme {

 public:

  EcalPreshowerNumberingScheme();
  ~EcalPreshowerNumberingScheme() override;
  uint32_t getUnitID(const EcalBaseNumber& baseNumber) const override ;

 private:

  int L3ax[3];
  int L3ay[3];
  int L3bx[1];
  int L3by[1];
  int L2ax[3];
  int L2ay[3];
  int L2bx[1];
  int L2by[1];
  int L1ax[26];
  int L1ay[26];
  int L1bx[1];
  int L1by[1];
  int L1cx[1];
  int L1cy[1];
  int L1dx[1];
  int L1dy[1];
  int L1ex[1];
  int L1ey[1];
  int L0ax[23];
  int L0ay[23];
  int L0bx[1];
  int L0by[1];
  int L0cx[1];
  int L0cy[1];

};

#endif
