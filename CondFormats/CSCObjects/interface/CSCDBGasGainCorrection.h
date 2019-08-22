#ifndef CSCDBGasGainCorrection_h
#define CSCDBGasGainCorrection_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iosfwd>
#include <vector>

class CSCDBGasGainCorrection {
public:
  CSCDBGasGainCorrection() {}
  ~CSCDBGasGainCorrection() {}

  struct Item {
    float gainCorr;

    COND_SERIALIZABLE;
  };

  typedef std::vector<Item> GasGainContainer;
  GasGainContainer gasGainCorr;

  const Item& item(int index) const { return gasGainCorr[index]; }
  float value(int index) const { return gasGainCorr[index].gainCorr; }

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const CSCDBGasGainCorrection& cscdb);

#endif
