#ifndef CSCNoiseMatrix_h
#define CSCNoiseMatrix_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCNoiseMatrix{
 public:
  CSCNoiseMatrix();
  ~CSCNoiseMatrix();
  
  struct Item{
    float elem33,elem34,elem35,elem44,elem45,elem46,
          elem55,elem56,elem57,elem66,elem67,elem77;
    std::string print() const;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;
  std::string print() const;
  
  typedef std::map< int,std::vector<Item> > NoiseMatrixMap;
  NoiseMatrixMap matrix;
};

#endif

