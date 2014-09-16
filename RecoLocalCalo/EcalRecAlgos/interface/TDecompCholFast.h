#ifndef TDecompCholFast_h
#define TDecompCholFast_h

#include "TDecompChol.h"

class TDecompCholFast : public TDecompChol {
  public:
    TDecompCholFast() {}
    
    void SetMatrixFast(const TMatrixDSym& a, double *data);
};

#endif
