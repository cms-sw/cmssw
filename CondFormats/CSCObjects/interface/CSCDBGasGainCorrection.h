#ifndef CSCDBGasGainCorrection_h
#define CSCDBGasGainCorrection_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>

class CSCDBGasGainCorrection{
 public:
  CSCDBGasGainCorrection();
  ~CSCDBGasGainCorrection();
  
  struct Item{
    float gainCorr;
  };

  // accessor to appropriate element...
  const Item & item(const CSCDetId & cscId, int strip, int wire) const;

  typedef std::vector<Item> GasGainContainer;

  GasGainContainer gasGainCorr;
};

#endif
