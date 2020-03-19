#ifndef CSCDBGains_h
#define CSCDBGains_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iosfwd>
#include <vector>

class CSCDBGains {
public:
  CSCDBGains() {}
  ~CSCDBGains() {}

  struct Item {
    short int gain_slope;

    COND_SERIALIZABLE;
  };
  int factor_gain;

  enum factors { FGAIN = 1000 };

  typedef std::vector<Item> GainContainer;
  GainContainer gains;

  const Item& item(int index) const { return gains[index]; }
  short int gain(int index) const { return gains[index].gain_slope; }
  int scale() const { return factor_gain; }

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const CSCDBGains& cscdb);

#endif
