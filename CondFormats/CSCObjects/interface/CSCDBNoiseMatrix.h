#ifndef CSCDBNoiseMatrix_h
#define CSCDBNoiseMatrix_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>

class CSCDBNoiseMatrix{
 public:
  CSCDBNoiseMatrix();
  ~CSCDBNoiseMatrix();
  
  struct Item{
    short int elem33,elem34,elem35,elem44,elem45,elem46,
              elem55,elem56,elem57,elem66,elem67,elem77;
  };
  int factor_noise;

  enum factors{FNOISE=1000};

  // accessor to appropriate element
  const Item & item(const CSCDetId & cscId, int strip) const;
  
  typedef std::vector<Item> NoiseMatrixContainer;

  NoiseMatrixContainer matrix;
};

#endif
