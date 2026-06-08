#ifndef HBHERUN3FLAGS_H
#define HBHERUN3FLAGS_H

#include <cstdint>
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"

class HBHERun3Flags {
public:
  HBHERun3Flags();
  ~HBHERun3Flags();

  bool repeatedADCblock(const QIE11DataFrame& digi, const int soi);
  bool isStuckADC(const QIE11DataFrame& digi);

  bool isBadCapId(const QIE11DataFrame& digi, const int soi, const uint32_t bunchCrossing);
  bool nonRotatingCapId(const QIE11DataFrame& digi);

private:
  static constexpr uint32_t stuckADC_min_ = 30;
  static constexpr int repeatedADCblock_min_ = 30;

  static constexpr short expCapIdInSOI_ = 1;
  static constexpr short nCapsQIE11_ = 4;
};

#endif
