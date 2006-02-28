#ifndef CSCMatrix_h
#define CSCMatrix_h

#include <vector>

class CSCMatrix{
 public:
  CSCMatrix();
  ~CSCMatrix();
  
  struct Item{
    float matrix_comp;
  };
  
  std::vector<Item> matrix;
};

#endif

