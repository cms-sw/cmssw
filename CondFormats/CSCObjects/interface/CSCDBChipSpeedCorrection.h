#ifndef CSCDBChipSpeedCorrection_h
#define CSCDBChipSpeedCorrection_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>

class CSCDBChipSpeedCorrection{
 public:
  CSCDBChipSpeedCorrection();
  ~CSCDBChipSpeedCorrection();
  
  struct Item{
    short int speedCorr;
  };
  int factor_speedCorr;

  /////change to the correct factor for you!
  enum factors{FCORR=100};

  // accessor to appropriate element ->should be chip !!!!
  const Item & item(const CSCDetId & cscId, int chip) const;

  typedef std::vector<Item> ChipSpeedContainer;

  ChipSpeedContainer chipSpeedCorr;
};

#endif
