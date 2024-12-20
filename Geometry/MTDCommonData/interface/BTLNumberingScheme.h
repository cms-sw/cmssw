#ifndef BTLNumberingScheme_h
#define BTLNumberingScheme_h

#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

class BTLNumberingScheme : public MTDNumberingScheme {
public:
  static constexpr uint32_t kBTLcrystalLevel = 9;
  static constexpr uint32_t kBTLmoduleLevel = 8;

  // to temporarily map V3 into V2-like input
  static constexpr std::array<uint32_t, BTLDetId::kRUPerTypeV2 * BTLDetId::kCrystalTypes> globalru2type{
      {1, 1, 2, 2, 3, 3}};
  static constexpr std::array<uint32_t, BTLDetId::kRUPerTypeV2 * BTLDetId::kCrystalTypes> globalru2ru{
      {1, 2, 1, 2, 1, 2}};

  BTLNumberingScheme();
  ~BTLNumberingScheme() override;
  uint32_t getUnitID(const MTDBaseNumber& baseNumber) const override;
};

#endif
