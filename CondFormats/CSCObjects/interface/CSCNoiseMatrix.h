#ifndef CSCNoiseMatrix_h
#define CSCNoiseMatrix_h

#include <vector>
#include <map>

class CSCNoiseMatrix{
 public:
  CSCNoiseMatrix();
  ~CSCNoiseMatrix();
  
  struct Item{
    float elem33,elem34,elem35,elem44,elem45,elem46,
          elem55,elem56,elem57,elem66,elem67,elem77;
  };

  const Item & item(int cscId, int strip) const;
  
  typedef std::map< int,std::vector<Item> > NoiseMatrixMap;
  NoiseMatrixMap matrix;
};

#endif

