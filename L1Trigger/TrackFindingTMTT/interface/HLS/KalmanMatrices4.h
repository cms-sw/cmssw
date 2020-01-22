/**
 * This defines the KF matrices and the operations performance on them.
 *
 *  All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 *
 * Author: Ian Tomalin
 */
 
#ifndef __KalmanMatrices4__
#define __KalmanMatrices4__

// Defines StateHLS & KFstate. Also defines finite bit integers & floats.
#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstate.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFconstants.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices4.h"
#else
#include "HLSutilities.h"
#include "KFstub.h"
#include "KFstate.h"
#include "KFconstants.h"
#include "KalmanMatrices.h"
#include "KalmanMatrices4.h"
#endif
 
#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

// Calculate matrix of derivatives of predicted stub coords w.r.t. helix params.

template <>
class MatrixH<4> {
public:
  enum {BH=KFstubN::BR+1};
  typedef ap_fixed<BH,BH>  TH;  // One extra bit, since "-r" can be -ve.
  typedef ap_ufixed<1,1>   T1;
  MatrixH(const KFstubN::TR& r) : _00(-r), _12(r),
                                           _01(1), _02(0), _03(0),
                                  _10(0),  _11(0),         _13(1) {}
public:
  TH       _00, _12;
  const T1      _01, _02, _03, 
           _10, _11,      _13;
};

// S = H * C

template <>
class MatrixS<4> {
public:
  enum {BH=MatrixH<4>::BH,
  // Calculate number of integer bits required for all non-zero elements of S.
  // (Assumes that some elements of C & H are zero and that all non-trivial elements of H have BH bits).
        BS00=MAX2(BH+KFstateN::BC00, KFstateN::BC01) - BODGE<4>::S,  // H00*C00 + H01*C10 + (H02*C20 + H03*C30 = zero).
        BS01=MAX2(BH+KFstateN::BC01, KFstateN::BC11) - BODGE<4>::S,  // H00*C01 + H01*C11 + (H02*C21 + H03*C31 = zero).
        BS12=MAX2(BH+KFstateN::BC22, KFstateN::BC23) - BODGE<4>::S,  // (H00*C02 + H01*C12 = zero) + H02*C22 + H03*C32.
	BS13=MAX2(BH+KFstateN::BC23, KFstateN::BC33) - BODGE<4>::S   // (H00*C03 + H01*C13 = zero) + H02*C23 + H03*C33.
       }; 
  typedef ap_fixed<B27,BS00>  TS00;
  typedef ap_fixed<B27,BS01>  TS01;
  typedef ap_fixed<B27,BS12>  TS12;
  typedef ap_fixed<B27,BS13>  TS13;
  typedef ap_fixed<BCORR,0>   T0;     // Neglect correlation between r-phi & r-z planes for now. 

public:

  MatrixS(const MatrixH<4>& H, const MatrixC<4>& C);

public:
  
  TS00 _00;
  TS01 _01;
  TS12 _12;
  TS13 _13;
  T0            _02, _03,
      _10, _11          ;
};

// Covariance matrix of helix params.

template <>
class MatrixC<4> {
public:
  typedef ap_ufixed<1,1>      T0; // HLS doesn't like zero bit variables.

  // Determine input helix coviaraiance matrix.
  MatrixC(const KFstate<4>& stateIn) :
             _00(stateIn.cov_00), _11(stateIn.cov_11), _22(stateIn.cov_22), _33(stateIn.cov_33), 
	     _01(stateIn.cov_01), _23(stateIn.cov_23), 
             _02(0), _03(0), _12(0), _13(0),
             _10(_01), _32(_23), _20(_02), _30(_03), _21(_12), _31(_13) 
  {
    _00[0] = 1;
    _11[0] = 1;
    _22[0] = 1;
    _33[0] = 1;
    _01[0] = 1;
    _23[0] = 1;
  }

  // Calculate output helix covariance matrix: C' = C - K*H*C = C - K*S.
  MatrixC(const MatrixC<4>& C, const MatrixK<4>& K, const MatrixS<4>& S);

public:
  // Elements that are finite
  KFstateN::TC00EX _00; 
  KFstateN::TC11EX _11; 
  KFstateN::TC22EX _22; 
  KFstateN::TC33EX _33; 
  KFstateN::TC01EX  _01; // (inv2R, phi0) -- other off-diagonal elements assumed negligeable.
  KFstateN::TC23EX  _23; // (tanL,  z0)   -- other off-diagonal elements assumed negligeable.
  // Elements that are zero.
  const T0 _02, _03, _12, _13;
  // Elements below the diagonal of this symmetric matrix.
  const KFstateN::TC01EX &_10;
  const KFstateN::TC23EX &_32;
  const T0 &_20, &_30, &_21, &_31;
};

// S(transpose) = C*H(transpose)

template <>
class MatrixS_transpose<4> {
public:
  typedef MatrixS<4>::TS00  TS00;
  typedef MatrixS<4>::TS01  TS01;
  typedef MatrixS<4>::TS12  TS12;
  typedef MatrixS<4>::TS13  TS13;
  typedef MatrixS<4>::T0    T0;
  MatrixS_transpose(const MatrixS<4>& S) : _00(S._00), _10(S._01), _21(S._12), _31(S._13),
					   _01(S._10), _11(S._11), _20(S._02), _30(S._03) {}
public:
  const TS00&  _00;
  const TS01&  _10;
  const TS12&  _21;
  const TS13&  _31;
  const T0&       _01,
                  _11,
             _20,
             _30     ;
};

// Covariance matrix of predicted residuals R = V + H*C*Ht = V + H*St.

template <>
class MatrixR<4> {
public:
  enum {BH=MatrixH<4>::BH,
	BS00=MatrixS<4>::BS00,
	BS01=MatrixS<4>::BS01,
	BS12=MatrixS<4>::BS12,
	BS13=MatrixS<4>::BS13,
        // Calculate number of integer bits required for elements of R.
        BR00 = MAX2(MatrixV::BVPP, MAX2(BH+BS00, BS01)) - BODGE<4>::R, // H00*St00 + H01*St10 + (H02*St20 + H03*St30 = zero)
	BR11 = MAX2(MatrixV::BVZZ, MAX2(BH+BS12, BS13)) - BODGE<4>::R, // (H10*St01 + H11*St11 = zero) + H12*St21 + H13*St31
	BR01 = 0                                                       // (H00*St01 + H01*St11 + H02*St21 + H03*St31 = zero)
       };  
  typedef ap_ufixed<B34,BR00>   TR00;
  typedef ap_ufixed<B34,BR11>   TR11;
  typedef ap_ufixed<BCORR,BR01> TR01;

public:
  MatrixR(const MatrixV& V, const MatrixH<4>& H, const MatrixS_transpose<4>& St);

public:
  TR00  _00;
  TR11  _11;
  TR01  _01;
  TR01& _10; // Matrix symmetric, so can use reference.
};

// Kalman gain matrix K = S(transpose)*R(inverse).

template <>
class MatrixK<4> {
public:
  enum {BS00=MatrixS<4>::BS00,
	BS01=MatrixS<4>::BS01,
	BS12=MatrixS<4>::BS12,
	BS13=MatrixS<4>::BS13,
        BIR00=MatrixInverseR<4>::BIR00,
        BIR11=MatrixInverseR<4>::BIR11,
        BIR01=MatrixInverseR<4>::BIR01,
        BK00=(BS00+BIR00) - BODGE<4>::K,  // St00*Rinv00 (+ St01*Rinv10 = zero)
        BK10=(BS01+BIR00) - BODGE<4>::K,  // St10*Rinv00 (+ St11*Rinv10 = zero)
        BK21=(BS12+BIR11) - BODGE<4>::K,  // (St20*Rinv01 = zero) + St21*Rinv11
        BK31=(BS13+BIR11) - BODGE<4>::K}; // (St30*Rinv01 = zero) + St31*Rinv11
  typedef ap_fixed<B35,BK00>  TK00;
  typedef ap_fixed<B35,BK10>  TK10;
  typedef ap_fixed<B35,BK21>  TK21;
  typedef ap_fixed<B35,BK31>  TK31;
  typedef ap_fixed<BCORR,0>   T0; // Neglect correlation between r-phi & r-z
  MatrixK(const MatrixS_transpose<4>& St, const MatrixInverseR<4>& RmatInv);
public:
  // Additional types used to cast this matrix to a lower precision one for updated helix param calculation.
  typedef ap_fixed<B27,BK00>  TK00_short;
  typedef ap_fixed<B27,BK10>  TK10_short;
  typedef ap_fixed<B27,BK21>  TK21_short;
  typedef ap_fixed<B27,BK31>  TK31_short;
public:
  TK00  _00;
  TK10  _10;
  TK21      _21;
  TK31      _31;
  T0        _01,
            _11,
        _20,
        _30    ;
};


// Hit residuals: res = m - H*x. 

template <>
class VectorRes<4> {
public:
  // Use higher granularity for residuals than for stubs.
  // BODGE<4>::RES should be slightly larger than BODGE_V as hits can be several sigma wrong.
  // Add one extra fractional bit relative to stub, to avoid additional precision loss.
  typedef ap_fixed<B18-BODGE<4>::RES+1,KFstubN::BPHI -BODGE<4>::RES> TRP;
  typedef ap_fixed<B18-BODGE<4>::RES+1,KFstubN::BZ1  -BODGE<4>::RES> TRZ;

public:
  VectorRes(const VectorM& m, const MatrixH<4>& H, const VectorX<4>& x);

public:
  TRP _0;
  TRZ _1;
};

// Vector of helix params.

template <>
class VectorX<4> {
public:
  // Determine input helix params.
  VectorX(const KFstate<4>& stateIn) : _0(stateIn.inv2R), _1(stateIn.phi0), _2(stateIn.tanL), _3(stateIn.z0) {} 

  // Calculate output helix params: x' = x + K*res
  VectorX(const VectorX<4>& x, const MatrixK<4>& K, const VectorRes<4>& res);

public:
  KFstateN::TR _0;
  KFstateN::TP _1;  
  KFstateN::TT _2;  
  KFstateN::TZ _3;  
};

#ifdef CMSSW_GIT_HASH
}

}
#endif

#endif
