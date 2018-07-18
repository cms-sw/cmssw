#ifndef BTLNumberingScheme_h
#define BTLNumberingScheme_h

#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"

class BTLNumberingScheme : public MTDNumberingScheme {
 public:
  BTLNumberingScheme();
  ~BTLNumberingScheme() override;
  uint32_t getUnitID(const MTDBaseNumber& baseNumber) const override ;
};

#endif
