#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"
#include <cmath>

HcalCholeskyMatrix::HcalCholeskyMatrix(int fId) : mId (fId)
{
   for(int cap = 0; cap != 4; cap++)
      for(int i = 0; i != 55; i++)
         cmatrix[cap][i] = 0;
}

float
HcalCholeskyMatrix::getValue(int capid, int i,int j) const
{
   if(i < j) return 0;
   int ii = i + 1;
   int jj = j + 1;
   float blah = (float)(cmatrix[capid][(ii*(ii-1)/2+jj)-1]);
   return blah/1000;
}

void
HcalCholeskyMatrix::setValue(int capid, int i, int j, float val)
{
   if(i < j) return;
   int ii = i + 1;
   int jj = j + 1;
   cmatrix[capid][(int)(ii*(ii-1)/2+jj)-1] = (signed short int)(floor)(val*10000);
}

