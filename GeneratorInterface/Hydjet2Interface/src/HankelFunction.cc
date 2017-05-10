/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#include <TMath.h>

#include "GeneratorInterface/Hydjet2Interface/interface/HankelFunction.h"

//compute Hankel function of zeroth order
enum {kNe = 2, kNCoeff = 9};
      
const double i0Coefficient[kNCoeff][kNe] = 
  {
    //coefficients to compute function I0
    {1.0,        0.39894228},
    {3.5156229,  0.01328592},
    {3.0899424,  0.00225319},
    {1.2067492, -0.00157565},
    {0.2659732,  0.00916281},
    {0.0360768, -0.02057706},
    {0.0045813,  0.02635537},
    {0.,        -0.01647633},
    {0.,         0.00392377}
  };

const double i1Coefficient[kNCoeff][kNe] =
  {
    //coefficients to compute function I1
    {0.5,         0.39894228},
    {0.87890594, -0.03988024},
    {0.51498869, -0.00362018},
    {0.15084934,  0.00163801},
    {0.02658733, -0.01031555},
    {0.00301532,  0.02282967},
    {0.00032411, -0.02895312},
    {0.,          0.01787654},
    {0.,         -0.00420059}
  };

const double k0Coefficient[kNCoeff][kNe] =
  {
    //coefficients to compute modified Hankel function of the zeroth order K0
    {-0.57721566, 1.25331414},   
    {0.42278420, -0.07832358}, 
    {0.23069756,  0.02189568}, 
    {0.03488590, -0.01062446}, 
    {0.00262698,  0.00587872}, 
    {0.00010750, -0.00251540}, 
    {0.0000074,   0.00053208}, 
    {0.,          0.        }, 
    {0.,          0.        }
  };

const double k1Coefficient[kNCoeff][kNe] =
  {
    //coefficients to compute modified Hankel function of the first order K1
    {1.0,          1.25331414},   
    {0.15443144,   0.23498619}, 
    {-0.67278579, -0.03655620}, 
    {-0.18156897,  0.01504268}, 
    {-0.01919402, -0.00780353}, 
    {-0.00110404,  0.00325614}, 
    {-0.00004686, -0.00068245}, 
    { 0.,          0.        }, 
    { 0.,          0.        }
  };

double BesselI0(double x) { 
  //  (C) Copr. 1986-92 Numerical Recipes Software +7.
  //compute Bessel function of zeroth order
  
  const double p1 = i0Coefficient[0][0];
  const double p2 = i0Coefficient[1][0];
  const double p3 = i0Coefficient[2][0];
  const double p4 = i0Coefficient[3][0];
  const double p5 = i0Coefficient[4][0];
  const double p6 = i0Coefficient[5][0];
  const double p7 = i0Coefficient[6][0];
  
  const double q1 = i0Coefficient[0][1];
  const double q2 = i0Coefficient[1][1];
  const double q3 = i0Coefficient[2][1];
  const double q4 = i0Coefficient[3][1];
  const double q5 = i0Coefficient[4][1];
  const double q6 = i0Coefficient[5][1];
  const double q7 = i0Coefficient[6][1];
  const double q8 = i0Coefficient[7][1];
  const double q9 = i0Coefficient[8][1];

  double i0 = 0.;

  if(TMath::Abs(x) < 3.75) {
    double y = (x / 3.75) * (x / 3.75);
    i0 =  p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7)))));
  } 
  else {
    double ax = TMath::Abs(x);
    double y = 3.75 / ax;
    i0 = (TMath::Exp(ax)/TMath::Sqrt(ax))*(q1 + y*(q2 + y*(q3 + y*(q4 + y*(q5 + y*(q6 + y*(q7 + y*(q8 + y*q9))))))));
  }
  
  return i0;
}

double BesselI1(double x) {
  //  (C) Copr. 1986-92 Numerical Recipes Software +7.
  //compute Bessel function of first order

  const double p1 = i1Coefficient[0][0];
  const double p2 = i1Coefficient[1][0];
  const double p3 = i1Coefficient[2][0];
  const double p4 = i1Coefficient[3][0];
  const double p5 = i1Coefficient[4][0];
  const double p6 = i1Coefficient[5][0];
  const double p7 = i1Coefficient[6][0];

  const double q1 = i1Coefficient[0][1];
  const double q2 = i1Coefficient[1][1];
  const double q3 = i1Coefficient[2][1];
  const double q4 = i1Coefficient[3][1];
  const double q5 = i1Coefficient[4][1];
  const double q6 = i1Coefficient[5][1];
  const double q7 = i1Coefficient[6][1];
  const double q8 = i1Coefficient[7][1];
  const double q9 = i1Coefficient[8][1];

  double i1 = 0.;

  if (TMath::Abs(x) < 3.75) {
    double y = (x / 3.75) * (x / 3.75);
    i1 = x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7))))));
  } 
  else {
    double ax = TMath::Abs(x);
    double y = 3.75/ax;
    i1 = (TMath::Exp(ax)/TMath::Sqrt(ax))*(q1 + y*(q2 + y*(q3 + y*(q4 + y*(q5 + y*(q6 + y*(q7 + y*(q8 + y*q9))))))));
    if(x < 0.) i1 = -i1;
  }

  return i1;
}

double HankelK0(double x) { 
  const double p1 = k0Coefficient[0][0];
  const double p2 = k0Coefficient[1][0];
  const double p3 = k0Coefficient[2][0];
  const double p4 = k0Coefficient[3][0];
  const double p5 = k0Coefficient[4][0];
  const double p6 = k0Coefficient[5][0];
  const double p7 = k0Coefficient[6][0];

  const double q1 = k0Coefficient[0][1];
  const double q2 = k0Coefficient[1][1];
  const double q3 = k0Coefficient[2][1];
  const double q4 = k0Coefficient[3][1];
  const double q5 = k0Coefficient[4][1];
  const double q6 = k0Coefficient[5][1];
  const double q7 = k0Coefficient[6][1];

  double k0 = 0.;
  if(x <= 2.0) {
    double y = x * x / 4.0;
    k0 = (-TMath::Log(x/2.0)*BesselI0(x)) + (p1 + y*(p2 + y*(p3 + y*(p4 + y*(p5 + y*(p6 + y*p7))))));
  } 
  else {
    double y = (2.0 / x);
    k0 = (TMath::Exp(-x)/TMath::Sqrt(x))*(q1 + y*(q2 + y*(q3 + y*(q4 + y*(q5 + y*(q6 + y*q7))))));
  }

  return k0;
}

double HankelK1(double x) { 
  //  (C) Copr. 1986-92 Numerical Recipes Software +7.
  // compute modified Hankel function of the first order
  const double p1 = k1Coefficient[0][0];
  const double p2 = k1Coefficient[1][0];
  const double p3 = k1Coefficient[2][0];
  const double p4 = k1Coefficient[3][0];
  const double p5 = k1Coefficient[4][0];
  const double p6 = k1Coefficient[5][0];
  const double p7 = k1Coefficient[6][0];

  const double q1 = k1Coefficient[0][1];
  const double q2 = k1Coefficient[1][1];
  const double q3 = k1Coefficient[2][1];
  const double q4 = k1Coefficient[3][1];
  const double q5 = k1Coefficient[4][1];
  const double q6 = k1Coefficient[5][1];
  const double q7 = k1Coefficient[6][1];

  double k1 = 0.;

  if(x <= 2.0) { 
    double y = x * x / 4.0;
    k1 = (TMath::Log(x/2.0)*BesselI1(x)) + (1.0/x)*(p1 + y*(p2 + y*(p3 + y*(p4 + y*(p5 + y*(p6 + y*p7))))));
  } 
  else {
    double y = 2.0 / x;
    k1 = (TMath::Exp(-x)/TMath::Sqrt(x))*(q1 + y*(q2 + y*(q3 + y*(q4 + y*(q5 + y*(q6 + y*q7))))));
  }

  return k1;
}

double HankelKn(int n, double x) {
  //  (C) Copr. 1986-92 Numerical Recipes Software +7.
  // compute modified Hankel function of the first order
  if(n < 2) throw "Bad argument n in Kn";

  double tox = 2.0 / x;
  double km = HankelK0(x);
  double k = HankelK1(x);
  double kp = 0.;
  
  for(int c = 1; c <= n-1; ++c) {
    kp = km + c * tox * k;
    km = k;
    k = kp;
  }

  return k;
}

