#ifndef CSCNoiseMatrix_h
#define CSCNoiseMatrix_h

#include <vector>
#include <map>

class CSCNoiseMatrix{
 public:
  CSCNoiseMatrix();
  ~CSCNoiseMatrix();
  
  struct Item{
    float tmp0_1,tmp0_2,tmp0_3,tmp0_4,tmp0_5,tmp0_6,tmp0_7,tmp0_8,tmp0_9,tmp0_10,tmp0_11,tmp0_12;
    float tmp1_1,tmp1_2,tmp1_3,tmp1_4,tmp1_5,tmp1_6,tmp1_7,tmp1_8,tmp1_9,tmp1_10,tmp1_11,tmp1_12;
  };
  
  std::map< int,std::vector<Item> > matrix;
};

#endif

