#ifndef HcalCholeskyMatrix_h
#define HcalCholeskyMatrix_h

#include <boost/cstdint.hpp>
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include <math.h>

class HcalCholeskyMatrix {
   public:
   HcalCholeskyMatrix(int fId=0);
 
   float getValue(int capid, int i,int j) const;// {return cmatrix[capid][i][j];}
   void setValue(int capid, int i, int j, float val);// {cmatrix[capid][i][j] = val;}

   uint32_t rawId () const {return mId;}

   private:
   signed short int cmatrix[4][55];
   uint32_t mId;
//   float cmatrix[4][10][10];
};
#endif
