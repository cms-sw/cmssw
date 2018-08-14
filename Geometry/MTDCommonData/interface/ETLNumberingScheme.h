#ifndef ETLNumberingScheme_h
#define ETLNumberingScheme_h

#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"

class ETLNumberingScheme : public MTDNumberingScheme {
 public:
  ETLNumberingScheme();
  ~ETLNumberingScheme() override;
  uint32_t getUnitID(const MTDBaseNumber& baseNumber) const override ;
};

#endif
