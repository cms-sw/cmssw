///=== This is the base class for the Kalman Combinatorial Filter track fit algorithm.

///=== All variable names & equations come from Fruhwirth KF paper
///=== http://dx.doi.org/10.1016/0168-9002%2887%2990887-4

///=== Written by: Ian Tomalin

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices5.h"
#else
#include "KalmanMatrices5.h"
#endif

#ifdef PRINT_SUMMARY
#include <iostream>
#endif

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

// Calculate matrix of derivatives of predicted stub coords w.r.t. helix params.

MatrixH<5>::THD MatrixH<5>::setH04(const KFstubN::TR& r) {
  static const MinusOneOverR calcMinusOneOverR; // Get -1/r   
  THD h04 = calcMinusOneOverR.get[r.to_uint() >> MinusOneOverR::BRED];
  return h04;
}

// Calculate S = H * C

MatrixS<5>::MatrixS(const MatrixH<5>& H, const MatrixC<5>& C) {
  _00 = H._00 * C._00 + H._01 * C._10 + H._02 * C._20 + H._03 * C._30 + H._04 * C._40;  
  _01 = H._00 * C._01 + H._01 * C._11 + H._02 * C._21 + H._03 * C._31 + H._04 * C._41;  
  _02 = H._00 * C._02 + H._01 * C._12 + H._02 * C._22 + H._03 * C._32 + H._04 * C._42;  
  _03 = H._00 * C._03 + H._01 * C._13 + H._02 * C._23 + H._03 * C._33 + H._04 * C._43;  
  _04 = H._00 * C._04 + H._01 * C._14 + H._02 * C._24 + H._03 * C._34 + H._04 * C._44;  
  _10 = H._10 * C._00 + H._11 * C._10 + H._12 * C._20 + H._13 * C._30 + H._14 * C._40;  
  _11 = H._10 * C._01 + H._11 * C._11 + H._12 * C._21 + H._13 * C._31 + H._14 * C._41;  
  _12 = H._10 * C._02 + H._11 * C._12 + H._12 * C._22 + H._13 * C._32 + H._14 * C._42;  
  _13 = H._10 * C._03 + H._11 * C._13 + H._12 * C._23 + H._13 * C._33 + H._14 * C._43;  
  _14 = H._10 * C._04 + H._11 * C._14 + H._12 * C._24 + H._13 * C._34 + H._14 * C._44;  

#ifdef PRINT_SUMMARY
  double s00 = H._00 * C._00 + H._01 * C._10 + H._02 * C._20 + H._03 * C._30 + H._04 * C._40;
  double s01 = H._00 * C._01 + H._01 * C._11 + H._02 * C._21 + H._03 * C._31 + H._04 * C._41;
  double s02 = H._00 * C._02 + H._01 * C._12 + H._02 * C._22 + H._03 * C._32 + H._04 * C._42;
  double s03 = H._00 * C._03 + H._01 * C._13 + H._02 * C._23 + H._03 * C._33 + H._04 * C._43;
  double s04 = H._00 * C._04 + H._01 * C._14 + H._02 * C._24 + H._03 * C._34 + H._04 * C._44;
  double s10 = H._10 * C._00 + H._11 * C._10 + H._12 * C._20 + H._13 * C._30 + H._14 * C._40;
  double s11 = H._10 * C._01 + H._11 * C._11 + H._12 * C._21 + H._13 * C._31 + H._14 * C._41;
  double s12 = H._10 * C._02 + H._11 * C._12 + H._12 * C._22 + H._13 * C._32 + H._14 * C._42;
  double s13 = H._10 * C._03 + H._11 * C._13 + H._12 * C._23 + H._13 * C._33 + H._14 * C._43;
  double s14 = H._10 * C._04 + H._11 * C._14 + H._12 * C._24 + H._13 * C._34 + H._14 * C._44;
  CHECK_AP::checkCalc("S00", _00, s00, 0.03);
  CHECK_AP::checkCalc("S01", _01, s01, 0.03);
  CHECK_AP::checkCalc("S02", _02, s02, 0.03);
  CHECK_AP::checkCalc("S03", _03, s03, 0.03);
  CHECK_AP::checkCalc("S04", _04, s04, 0.03);
  CHECK_AP::checkCalc("S10", _10, s10, 0.03);
  CHECK_AP::checkCalc("S11", _11, s11, 0.03);
  CHECK_AP::checkCalc("S12", _12, s12, 0.03);
  CHECK_AP::checkCalc("S13", _13, s13, 0.03);
  CHECK_AP::checkCalc("S14", _14, s14, 0.03);
#endif
}

// Calculate covariance matrix of predicted residuals R = V + H*C*Ht = V + H*St.

MatrixR<5>::MatrixR(const MatrixV& V, const MatrixH<5>& H, const MatrixS_transpose<5>& St) : 
  _10(_01) 
{
  _00 = V._00 + (H._00*St._00 + H._01*St._10 + H._02*St._20 + H._03*St._30 + H._04*St._40);
  _01 = V._01 + (H._00*St._01 + H._01*St._11 + H._02*St._21 + H._03*St._31 + H._04*St._41);
  // R._10 // Matrix symmetric so don't need to calculate this element.
  _11 = V._11 + (H._10*St._01 + H._11*St._11 + H._12*St._21 + H._13*St._31 + H._14*St._41);

#ifdef PRINT_SUMMARY
  double r00 = V._00 + (H._00*St._00 + H._01*St._10 + H._02*St._20 + H._03*St._30 + H._04*St._40);
  double r01 = V._01 + (H._00*St._01 + H._01*St._11 + H._02*St._21 + H._03*St._31 + H._04*St._41);
  double r11 = V._11 + (H._10*St._01 + H._11*St._11 + H._12*St._21 + H._13*St._31 + H._14*St._41);
  CHECK_AP::checkCalc("R00", _00, r00);
  CHECK_AP::checkCalc("R01", _01, r01);
  CHECK_AP::checkCalc("R11", _11, r11);
#endif
}

// Kalman gain matrix K = S*R(inverse).

MatrixK<5>::MatrixK(const MatrixS_transpose<5>& St, const MatrixInverseR<5>& RmatInv) {
  _00 =  St._00 * RmatInv._00 + St._01 * RmatInv._10;
  _10 =  St._10 * RmatInv._00 + St._11 * RmatInv._10;
  _20 =  St._20 * RmatInv._00 + St._21 * RmatInv._10;
  _30 =  St._30 * RmatInv._00 + St._31 * RmatInv._10;
  _40 =  St._40 * RmatInv._00 + St._41 * RmatInv._10;
  _01 =  St._00 * RmatInv._01 + St._01 * RmatInv._11;
  _11 =  St._10 * RmatInv._01 + St._11 * RmatInv._11;
  _21 =  St._20 * RmatInv._01 + St._21 * RmatInv._11;
  _31 =  St._30 * RmatInv._01 + St._31 * RmatInv._11;
  _41 =  St._30 * RmatInv._01 + St._41 * RmatInv._11;

#ifdef PRINT_SUMMARY
  double k00 =  double(St._00) * double(RmatInv._00) + double(St._01) * double(RmatInv._10);
  double k10 =  double(St._10) * double(RmatInv._00) + double(St._11) * double(RmatInv._10);
  double k20 =  double(St._20) * double(RmatInv._00) + double(St._21) * double(RmatInv._10);
  double k30 =  double(St._30) * double(RmatInv._00) + double(St._31) * double(RmatInv._10);
  double k40 =  double(St._40) * double(RmatInv._00) + double(St._41) * double(RmatInv._10);
  double k01 =  double(St._00) * double(RmatInv._01) + double(St._01) * double(RmatInv._11);
  double k11 =  double(St._10) * double(RmatInv._01) + double(St._11) * double(RmatInv._11);
  double k21 =  double(St._20) * double(RmatInv._01) + double(St._21) * double(RmatInv._11);
  double k31 =  double(St._30) * double(RmatInv._01) + double(St._31) * double(RmatInv._11);
  double k41 =  double(St._40) * double(RmatInv._01) + double(St._41) * double(RmatInv._11);
  CHECK_AP::checkCalc("K00", _00, k00, 0.001);
  CHECK_AP::checkCalc("K10", _10, k10, 0.001);
  CHECK_AP::checkCalc("K20", _20, k20, 0.001);
  CHECK_AP::checkCalc("K30", _30, k30, 0.001);
  CHECK_AP::checkCalc("K40", _40, k40, 0.001);
  CHECK_AP::checkCalc("K01", _01, k01, 0.001);
  CHECK_AP::checkCalc("K11", _11, k11, 0.001);
  CHECK_AP::checkCalc("K21", _21, k21, 0.001);
  CHECK_AP::checkCalc("K31", _31, k31, 0.001);
  CHECK_AP::checkCalc("K41", _41, k41, 0.001);
#endif
}

// Hit residuals: res = m - H*x. 

VectorRes<5>::VectorRes(const VectorM& m, const MatrixH<5>& H, const VectorX<5>& x) {
  _0 = m._0 - (H._00 * x._0 + H._01 * x._1 + H._02 * x._2 + H._03 * x._3 + H._04 * x._4);  
  _1 = m._1 - (H._10 * x._1 + H._11 * x._1 + H._12 * x._2 + H._13 * x._3 + H._14 * x._4);  
#ifdef PRINT_SUMMARY
  double r0 =  double(m._0) - (double(H._00) * double(x._0) + double(H._01) * double(x._1) + 
                               double(H._02) * double(x._2) + double(H._03) * double(x._3) +
			       double(H._04) * double(x._4));
  double r1 =  double(m._1) - (double(H._10) * double(x._0) + double(H._11) * double(x._1) + 
                               double(H._12) * double(x._2) + double(H._13) * double(x._3) +
			       double(H._14) * double(x._4));
  CHECK_AP::checkCalc("RES0", _0, r0, 0.1, 0.1);
  CHECK_AP::checkCalc("RES1", _1, r1, 0.1, 0.1);
#endif
}

// Calculate output helix params: x' = x + K*res

VectorX<5>::VectorX(const VectorX<5>& x, const MatrixK<5>& K, const VectorRes<5>& res) {
  typedef MatrixK<5>::TK00_short TK00_short;
  typedef MatrixK<5>::TK10_short TK10_short;
  typedef MatrixK<5>::TK21_short TK21_short;
  typedef MatrixK<5>::TK31_short TK31_short;
  typedef MatrixK<5>::TK40_short TK40_short;
  typedef MatrixK<5>::T0         T0;
  _0 = x._0 + KFstateN::TR(TK00_short(K._00) * res._0 + T0        (K._01) * res._1);
  _1 = x._1 + KFstateN::TP(TK10_short(K._10) * res._0 + T0        (K._11) * res._1);
  _2 = x._2 + KFstateN::TT(T0        (K._20) * res._0 + TK21_short(K._21) * res._1);
  _3 = x._3 + KFstateN::TZ(T0        (K._30) * res._0 + TK31_short(K._31) * res._1);
  _4 = x._4 + KFstateN::TD(TK40_short(K._40) * res._0 + T0        (K._41) * res._1);
}


// Calculate output helix covariance matrix: C' = C - K*H*C = C - K*S.

MatrixC<5>::MatrixC(const MatrixC<5>& C, const MatrixK<5>& K, const MatrixS<5>& S) :
  _02(0), _03(0), _12(0), _13(0), _42(0), _43(0),
  _10(_01), _32(_23), _40(_04), _41(_14), _20(_02), _30(_03), _21(_12), _31(_13), _24(_42), _34(_43)
{
  // Covariance matrix is symmetric & some elements can be neglected.
  _00 =  C._00 - KFstateN::TC00EX(K._00 * S._00 + K._01 * S._10);
  _11 =  C._11 - KFstateN::TC11EX(K._10 * S._01 + K._11 * S._11);
  _22 =  C._22 - KFstateN::TC22EX(K._20 * S._02 + K._21 * S._12);
  _33 =  C._33 - KFstateN::TC33EX(K._30 * S._03 + K._31 * S._13);
  _44 =  C._44 - KFstateN::TC44EX(K._40 * S._04 + K._41 * S._14);
  _01 =  C._01 - KFstateN::TC01EX(K._00 * S._01 + K._01 * S._11);
  _23 =  C._23 - KFstateN::TC23EX(K._20 * S._03 + K._21 * S._13);
  _04 =  C._04 - KFstateN::TC04EX(K._00 * S._04 + K._01 * S._14);
  _14 =  C._14 - KFstateN::TC14EX(K._10 * S._04 + K._11 * S._14);

#ifdef PRINT_SUMMARY
  double c00new = double(C._00) - (double(K._00) * double(S._00) + double(K._01) * double(S._10));
  double c11new = double(C._11) - (double(K._10) * double(S._01) + double(K._11) * double(S._11));
  double c22new = double(C._22) - (double(K._20) * double(S._02) + double(K._21) * double(S._12));
  double c33new = double(C._33) - (double(K._30) * double(S._03) + double(K._31) * double(S._13));
  double c44new = double(C._44) - (double(K._40) * double(S._04) + double(K._41) * double(S._14));
  double c01new = double(C._01) - (double(K._00) * double(S._01) + double(K._01) * double(S._11));
  double c23new = double(C._23) - (double(K._20) * double(S._03) + double(K._21) * double(S._13));
  double c04new = double(C._04) - (double(K._00) * double(S._04) + double(K._01) * double(S._14));
  double c14new = double(C._14) - (double(K._10) * double(S._04) + double(K._11) * double(S._14));
  CHECK_AP::checkCalc("C00_new", _00, c00new, 0.03);
  CHECK_AP::checkCalc("C11_new", _11, c11new, 0.03);
  CHECK_AP::checkCalc("C22_new", _22, c22new, 0.03);
  CHECK_AP::checkCalc("C33_new", _33, c33new, 0.03);
  CHECK_AP::checkCalc("C44_new", _44, c44new, 0.03);
  CHECK_AP::checkCalc("C01_new", _01, c01new, 0.03);
  CHECK_AP::checkCalc("C23_new", _23, c23new, 0.03);
  CHECK_AP::checkCalc("C04_new", _04, c04new, 0.03);
  CHECK_AP::checkCalc("C14_new", _14, c14new, 0.03);
  CHECK_AP::checkDet("C_new(rphi)",_00,_11,_44, _01, _04, _14);
  CHECK_AP::checkDet("C_new(rz)"  ,_22,_33,_23);
#endif
}

#ifdef CMSSW_GIT_HASH
}

}
#endif

