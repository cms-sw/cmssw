#ifndef CSCNoiseMatrix_h
#define CSCNoiseMatrix_h

#include <vector>
#include <map>

class CSCNoiseMatrix{
 public:
  CSCNoiseMatrix();
  ~CSCNoiseMatrix();
  
  struct Item{
    float elem1,elem2,elem3,elem4,elem5,elem6,elem7,elem8,elem9,elem10,elem11,elem12;
  };
  
  std::map< int,std::vector<Item> > matrix;
};

#endif

