#ifndef CSCDBNoiseMatrix_h
#define CSCDBNoiseMatrix_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iosfwd>
#include <vector>

class CSCDBNoiseMatrix {
public:
  CSCDBNoiseMatrix() {}
  ~CSCDBNoiseMatrix() {}

  struct Item {
    short int elem33, elem34, elem35, elem44, elem45, elem46, elem55, elem56, elem57, elem66, elem67, elem77;

    COND_SERIALIZABLE;
  };
  int factor_noise;

  enum factors { FNOISE = 1000 };

  typedef std::vector<Item> NoiseMatrixContainer;
  NoiseMatrixContainer matrix;

  const Item& item(int index) const { return matrix[index]; }
  int scale() const { return factor_noise; }

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const CSCDBNoiseMatrix& cscdb);

#endif
