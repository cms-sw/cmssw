#ifndef HcalCovarianceMatrix_h
#define HcalCovarianceMatrix_h

#include <boost/cstdint.hpp>
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include <math.h>

class HcalCovarianceMatrix {
   public:
   HcalCovarianceMatrix(int fId=0);
 
   float getValue(int capid, int i,int j) const {return covariancematrix[capid][i][j];}
   void setValue(int capid, int i, int j, float val) {covariancematrix[capid][i][j] = val;}

   uint32_t rawId () const {return mId;}

   private:
   uint32_t mId;
   float covariancematrix[4][10][10];
};
#endif
