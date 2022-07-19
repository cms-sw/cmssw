#ifndef BTLNumberingScheme_h
#define BTLNumberingScheme_h

#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"

class BTLNumberingScheme : public MTDNumberingScheme {
public:
  static constexpr uint32_t kBTLcrystalLevel = 9;
  static constexpr uint32_t kBTLmoduleLevel = 8;

  BTLNumberingScheme();
  ~BTLNumberingScheme() override;
  uint32_t getUnitID(const MTDBaseNumber& baseNumber) const override;
};

#endif
