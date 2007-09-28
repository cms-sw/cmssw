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
  ~EcalPreshowerNumberingScheme();
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const ;
  void findXY(const int& layer, const int& waf, int& x, int& y) const;

private:

  int iquad_max[40];
  int iquad_min[40];
  int Ntot[40];
  int Ncols[40];

};

#endif
