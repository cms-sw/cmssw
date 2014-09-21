#ifndef HcalCholeskyMatrix_h
#define HcalCholeskyMatrix_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <boost/cstdint.hpp>
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"

class HcalCholeskyMatrix {
   public:
   explicit HcalCholeskyMatrix(int fId=0);
 
   float getValue(int capid, int i,int j) const;// {return cmatrix[capid][i][j];}
   void setValue(int capid, int i, int j, float val);// {cmatrix[capid][i][j] = val;}

   uint32_t rawId () const {return mId;}

   private:

   static int findIndex(int i, int j);
   //Due to a bug in ROOT versions before 5.34.20
   // only the first 4 elements of the array were 
   // ever stored. The previous array dimensions were
   //signed short int cmatrix[4][55];
   signed short int cmatrix[4];
   uint32_t mId;

 COND_SERIALIZABLE;
};
#endif
