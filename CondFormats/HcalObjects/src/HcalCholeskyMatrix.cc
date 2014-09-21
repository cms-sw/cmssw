#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"
#include <cmath>

//Due to a bug in ROOT versions before 5.34.20  only the first 4 elements of the
// cmatrix array were ever stored. To reproduce the behavior, we make it appear that
// the matrix returns 0 for any other index request. Similarly, only the first
// 4 elements can be set.
HcalCholeskyMatrix::HcalCholeskyMatrix(int fId) : cmatrix{0,0,0,0},mId (fId)
{
}

inline int 
HcalCholeskyMatrix::findIndex(int i, int j) {
  int ii = i + 1;
  int jj = j + 1;
  return (ii*(ii-1)/2+jj)-1;
}

float
HcalCholeskyMatrix::getValue(int capid, int i,int j) const
{
   if(i < j) return 0.f;
   if(capid != 0) {return 0.f;}
   auto index = findIndex(i,j);
   if(index > 3) {
     return 0.f;
   }
   float blah = static_cast<float>(cmatrix[index]);
   return blah/1000.;
}

void
HcalCholeskyMatrix::setValue(int capid, int i, int j, float val)
{
   if(i < j) return;
   auto index = findIndex(i,j);
   if(capid == 0 and index < 4) {
     cmatrix[index] = static_cast<signed short int>((floor)(val*10000));
   }
}

