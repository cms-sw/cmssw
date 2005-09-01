// jetCalibV1Param.h 
//
// Decription
// -----------
// This header files contains the parameters of the jetCalibV1
// parameterizations.
//
// History
// -------
// 27-Sep-04   R. Harris  Initial Version
//
// Parameterization of the reconstructed jet response vs. gen jet Pt for |eta|<1.0
float ptHiPar[2]={9.94981e-02,-8.65615e-03};  // Parameters at gen Pt > ptCross 
float ptLoPar[2]={3.63558e-02, 1.75213e-01};  // Parameters at gen Pt < ptCross
float ptCross=44.;                            // The cross point of the two paramterizations in Pt.
float responseCross=0.67;                     // The value of the response at the cross point
//float fitLoPt=24.;                          //The lowest Pt value included in the fit
float extrapLoPt=10.;                         //The lowest Pt value to extrapolate to
float fitHiPt=4000.;                          // The highest Pt value included in the fit
float ptParam(float);                       // Declaration of parameterization vs pt

// Parameterizations of the response vs reconstructed |eta| relative to the resonse for |eta|<1.0.
// Independent parameterizations for each pt interval: pt0 to pt8.
// Parameterizations are polynomials of varying degree.
// 25 < Rec Pt < 30, 5th order polynomial
float etaParPt0[9]={1.06750e+00, 1.61281e-01, -8.52704e-01, 6.23863e-01, -1.53308e-01, 1.21999e-02, 0., 0., 0.};
float etaMaxPt0=4.9;

// 30 < Rec Pt < 40, 5th order polynomial
float etaParPt1[9]={1.07843e+00, 7.74872e-03, -5.34065e-01, 4.32094e-01, -1.09486e-01, 8.86472e-03, 0., 0., 0.};
float etaMaxPt1=5.2;

// 40 < Rec Pt < 60, 5th order polynomial
float etaParPt2[9]={1.07381e+00, -1.09056e-01, -2.01326e-01, 2.09611e-01, -5.45622e-02, 4.31300e-03, 0., 0., 0.};
float etaMaxPt2=4.7;

// 60 < Rec Pt < 120, 4th order polynomial
float etaParPt3[9]={1.08195e+00, -2.66375e-01, 1.41413e-01, -1.38969e-02, 4.63453e-05, 0., 0., 0., 0.};
float etaMaxPt3=3.6;

// 120 < Rec Pt < 250, 8th order polynomial
float etaParPt4[9]={1.05782e+00, -8.23542e-02, -9.34968e-01, 3.34166e+00, -4.94256e+00, 3.73568e+00, -1.49995e+00, 3.04343e-01, -2.45457e-02};
float etaMaxPt4=3.3;

// 250 < Rec Pt < 500, 6th order polynomial
float etaParPt5[9]={1.03183e+00, -1.83845e-01,  5.25477e-01, -8.47939e-01,  6.31875e-01, -2.07102e-01,  2.45602e-02, 0., 0.};
float etaMaxPt5=3.3;

// 500 < Rec Pt < 1000, 3rd order polynomial
float etaParPt6[9]={1.00234e+00,  6.23692e-02, -1.67730e-01, 8.09708e-02, 0., 0., 0., 0., 0.};
float etaMaxPt6=1.8;

// 1000 < Rec Pt < 2000, 5th order polynomial
float etaParPt7[9]={ 9.88800e-01, 2.15133e-01, -9.72889e-01, 1.71033e+00, -1.28386e+00, 3.38504e-01, 0., 0., 0.};
float etaMaxPt7=1.7;

// 2000 < Rec Pt < 4000, 3rd order polynomial
float etaParPt8[9]={ 1.01643e+00, -1.14347e-01, 2.04919e-01, -1.15234e-01, 0., 0., 0., 0., 0.};
float etaMaxPt8=1.5;

// The middle of the rec pt intervals
float meanPt[9]={27.5, 35.0, 50.0, 90.0, 185.0, 375.0, 750., 1500.0, 3000.0};

float etaParam(float , float);   //Declaration of eta parameterization
float polynomial(float* , float); //Declaration of polynomial
