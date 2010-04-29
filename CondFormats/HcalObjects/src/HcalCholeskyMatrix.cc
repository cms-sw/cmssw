#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"

HcalCholeskyMatrix::HcalCholeskyMatrix(int fId) : mId (fId)
{
//   for(int cap = 0; cap != 4; cap++)
//      for(int i = 0; i != 10; i++)
//         for(int j = 0; j != 10; j++)
//         cmatrix[cap][i][j] = 0;

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
   return (float)(cmatrix[capid][(ii*(ii-1)/2+jj)-1]);
}

void
HcalCholeskyMatrix::setValue(int capid, int i, int j, float val)
{
   if(i < j) return;
   int ii = i + 1;
   int jj = j + 1;
   cmatrix[capid][(int)(ii*(ii-1)/2+jj)-1] = val;
}


/*
void
HcalCholeskyMatrix::makeNoise(int m, float xr[], HcalPedestal& ADCped)
{
   double x1, x2, w, y1, y2;
   double z1[10], z2[10], X[10], Xr[10];
    
   for(int i=0; i!=10; i++){
      X[i] = 0;
      Xr[i] = 0;
   }

//............Random Gaussian Number Generator and assigned to four-vector z.
//............(creates two by using polar form as faster than calling trig function lib for sin/cos)....

   for(int i=0; i!=10; i++)
   {
      do{
         x1 = 2.0 * ranf() - 1.0;
         x2 = 2.0 * ranf() - 1.0;
         w = (x1*x1) + (x2*x2);
      } while ( w >= 1.0 );
      
      w = sqrt( (-2.0 * log( w ) ) / w );
      y1 = x1 * w;
      y2 = x2 * w; 
      z1[i] = y1;
      z2[i] = y2;
   }

   for(int i = 0; i != 10; i++){
      for(int j = 0; j != 10; j++){
         X[i] += cmatrix[m][i][j] * z1[j] * 0;
      }
   }

   for(int i = 0; i != 10; i++){
      X[i] += ADCped.getValue((i+m)%4);
   }

   for(int i = 0; i != 10; i++){
      if(X[i] < 0) X[i] = 0.01;
      xr[i] = round(X[i]);
   }

   return;
}

float
HcalCholeskyMatrix::round(float x) //also check for ADCs less than zero
{
        double k = 0;
        k = fmod(x, 1.0);
        if(k < 0.5)
        {
           return floor(x);
        }
        else
        {
           return ceil(x);
        }
}

float
HcalCholeskyMatrix::ranf()
{
   return ((float)random()/(1.0+(float)RAND_MAX));
}*/
