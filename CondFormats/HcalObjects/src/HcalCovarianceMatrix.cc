#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrix.h"

HcalCovarianceMatrix::HcalCovarianceMatrix(int fId) : mId (fId)
{
   for(int cap = 0; cap != 4; cap++)
      for(int i = 0; i != 10; i++)
         for(int j = 0; j != 10; j++)
         covariancematrix[cap][i][j] = 0;
}

