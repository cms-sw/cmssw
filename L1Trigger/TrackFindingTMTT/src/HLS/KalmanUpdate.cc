/**
 * This is the top-level HLS function, which updates a helix state by adding a stub to it.
 * N.B. It therefore can't use the Settings class or any external libraries! Nor can it be a C++ class.
 *
 * All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 * 
 * Author: Ian Tomalin
 */

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanUpdate.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFconstants.h"
#else
#include "KalmanUpdate.h"
#include "KalmanMatrices.h"
#include "HLSutilities.h"
#include "KFconstants.h"
#endif

#ifdef PRINT_SUMMARY
#include <iostream>
#endif

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

//--- Explicit instantiation required for all non-specialized templates, to allow them to be implemented 
//--- in .cc files.

template void kalmanUpdate(const KFstubC& stub, const KFstate<4>& stateIn, KFstate<4>& stateOut, KFselect<4>& selectOut);

template void kalmanUpdate(const KFstubC& stub, const KFstate<5>& stateIn, KFstate<5>& stateOut, KFselect<5>& selectOut);

template void calcDeltaChi2(const VectorRes<4>& res, const MatrixInverseR<4>& Rinv, TCHI_INT& dChi2_phi, TCHI_INT& dChi2_z);

template void calcDeltaChi2(const VectorRes<5>& res, const MatrixInverseR<5>& Rinv, TCHI_INT& dChi2_phi, TCHI_INT& dChi2_z);

//=== Add stub to old KF helix state to get new KF helix state.

template <unsigned int NPAR>
void kalmanUpdate(const KFstubC& stub, const KFstate<NPAR>& stateIn, KFstate<NPAR>& stateOut, KFselect<NPAR>& selectOut) {

  stateOut.cBin_ht = stateIn.cBin_ht;
  stateOut.mBin_ht = stateIn.mBin_ht;
  stateOut.layerID = stateIn.layerID;
  stateOut.nSkippedLayers = stateIn.nSkippedLayers;
  stateOut.hitPattern = stateIn.hitPattern;
  stateOut.trackID = stateIn.trackID;
  stateOut.eventID = stateIn.eventID;
  stateOut.phiSectID = stateIn.phiSectID;
  stateOut.etaSectID = stateIn.etaSectID;
  stateOut.etaSectZsign = stateIn.etaSectZsign;
  stateOut.valid = (stub.valid && stateIn.valid);

#ifdef PRINT_SUMMARY
  static bool first = true;
  if (first) {
    first = false;
    std::cout<<std::endl<<"KF HLS bodge bits: V="<<BODGE<NPAR>::V<<" S="<<BODGE<NPAR>::S<<" R="<<BODGE<NPAR>::R<<" IR="<<BODGE<NPAR>::IR<<" DET="<<BODGE<NPAR>::DET<<" K="<<BODGE<NPAR>::K<<" RES="<<BODGE<NPAR>::RES<<" CHI2="<<BODGE<NPAR>::CHI2<<std::endl<<std::endl;
  }
#endif

#ifdef PRINT
  std::cout<<"KalmanUpdate call: layerID="<<stateIn.layerID<<" nSkipped="<<stateIn.nSkippedLayers<<std::endl;
#endif

  // Store vector of stub coords.
  VectorM m(stub.phiS, stub.z);

  // Store covariance matrix of stub coords.
  MatrixV V(stub.r, stub.z, stateIn.inv2R, stateIn.tanL, stateIn.mBin_ht);

  // Store vector of input helix params.
  VectorX<NPAR> x(stateIn);

  // Store covariance matrix of input helix params.
  MatrixC<NPAR> C(stateIn);

  // Calculate matrix of derivatives of predicted stub coords w.r.t. helix params.
  MatrixH<NPAR> H(stub.r);

  // Calculate S = H*C, and its transpose St, which is equal to C*H(transpose).
  MatrixS<NPAR>           S(H, C);
  MatrixS_transpose<NPAR> St(S);

  // Calculate covariance matrix of predicted residuals R = V + H*C*Ht = V + H*St, and its inverse.
  // (Call this Rmat instead of R to avoid confusion with track radius).
  MatrixR<NPAR>  Rmat(V, H, St);
  MatrixInverseR<NPAR> RmatInv(Rmat);

  // Calculate Kalman gain matrix * determinant(R): K = S*R(inverse)
  MatrixK<NPAR> K(St, RmatInv);

  // Calculate hit residuals.
  VectorRes<NPAR> res(m, H, x); 

  // Calculate output helix params & their covariance matrix.
  VectorX<NPAR> x_new(x, K, res);
  MatrixC<NPAR> C_new(C, K, S);
 
  /*
  // Useful to debug C matrices with negative determinants, by fully recalculating them in double precision.
  double s00 = double(H._00) * double(C._00) + double(H._01) * double(C._10) + double(H._02) * double(C._20) + double(H._03) * double(C._30);
  double s01 = double(H._00) * double(C._01) + double(H._01) * double(C._11) + double(H._02) * double(C._21) + double(H._03) * double(C._31);
  double s02 = double(H._00) * double(C._02) + double(H._01) * double(C._12) + double(H._02) * double(C._22) + double(H._03) * double(C._32);
  double s03 = double(H._00) * double(C._03) + double(H._01) * double(C._13) + double(H._02) * double(C._23) + double(H._03) * double(C._33);
  double s10 = double(H._10) * double(C._00) + double(H._11) * double(C._10) + double(H._12) * double(C._20) + double(H._13) * double(C._30);
  double s11 = double(H._10) * double(C._01) + double(H._11) * double(C._11) + double(H._12) * double(C._21) + double(H._13) * double(C._31);
  double s12 = double(H._10) * double(C._02) + double(H._11) * double(C._12) + double(H._12) * double(C._22) + double(H._13) * double(C._32);
  double s13 = double(H._10) * double(C._03) + double(H._11) * double(C._13) + double(H._12) * double(C._23) + double(H._13) * double(C._33);
  double st00 = s00;
  double st10 = s01;
  double st20 = s02;
  double st30 = s03;
  double st01 = s10;
  double st11 = s11;
  double st21 = s12;
  double st31 = s13;  
  double r00 = double(V._00) + double(H._00)*(st00) + double(H._01)*(st10) + double(H._02)*(st20) + double(H._03)*(st30);
  double r11 = double(V._11) + double(H._10)*(st01) + double(H._11)*(st11) + double(H._12)*(st21) + double(H._13)*(st31);
  double rinv00 = 1./r00;
  double rinv11 = 1./r11;
  double k00 =  (st00)*rinv00; 
  double k10 =  (st10)*rinv00;
  double k20 =  (st20)*rinv00;
  double k30 =  (st30)*rinv00;
  double k01 =  (st01)*rinv11;
  double k11 =  (st11)*rinv11;
  double k21 =  (st21)*rinv11;
  double k31 =  (st31)*rinv11;
  double c22 =  double(C._22) - (k20 * (s02) + k21 * (s12));
  double c33 =  double(C._33) - (k30 * (s03) + k31 * (s13));
  double c23 =  double(C._23) - (k20 * (s03) + k21 * (s13));
  std::cout<<"recalc C new: TT="<<c22<<" ZZ="<<c33<<" TZ="<<c23<<std::endl;
  CHECK_AP::checkDet("recalc_rz",c22,c33,c23); 
 */

  // Calculate increase in chi2 (in r-phi & r-z) from adding new stub: delta(chi2) = res(transpose) * R(inverse) * res
  TCHI_INT deltaChi2_phi = 0, deltaChi2_z = 0;
  calcDeltaChi2(res, RmatInv, deltaChi2_phi, deltaChi2_z);
  TCHI_INT chi2_phi = stateIn.chiSquaredRphi + deltaChi2_phi;
  TCHI_INT chi2_z   = stateIn.chiSquaredRz   + deltaChi2_z;
  // Truncate chi2 to avoid overflow.
  static const TCHI_INT MAX_CHI2 = (1 << KFstateN::BCHI) - 1;
  if (chi2_phi > MAX_CHI2) chi2_phi = MAX_CHI2;
  if (chi2_z   > MAX_CHI2) chi2_z   = MAX_CHI2;
  stateOut.chiSquaredRphi = chi2_phi;
  stateOut.chiSquaredRz = chi2_z;
  
  stateOut.inv2R = x_new._0;
  stateOut.phi0  = x_new._1;
  stateOut.tanL  = x_new._2;
  stateOut.z0    = x_new._3;
  stateOut.cov_00 = C_new._00;
  stateOut.cov_11 = C_new._11;
  stateOut.cov_22 = C_new._22;
  stateOut.cov_33 = C_new._33;
  stateOut.cov_01 = C_new._01;
  stateOut.cov_23 = C_new._23;

  // Check if output helix passes cuts.
  // (Copied from Maxeller code KFWorker.maxj)
  ap_uint<3> nStubs = stateIn.layerID - stateIn.nSkippedLayers; // Number of stubs on state including current one.

  // IRT - feed in test helix params to debug cases seen in QuestaSim. (1/2r, phi, tanl, z0)
  //x_new._0 = float(-8163)/float(1 << (B18 - KFstateN::BH0));
  //x_new._1 = float(-57543)/float(1 << (B18 - KFstateN::BH1));
  //x_new._2 = float(4285)/float(1 << (B18 - KFstateN::BH2));
  //x_new._3 = float(-7652)/float(1 << (B18 - KFstateN::BH3));

  KFstateN::TZ   cut_z0          = z0Cut[nStubs];
  KFstateN::TZ   cut_z0_minus    = z0CutMinus[nStubs];
  KFstateN::TR   cut_inv2R       = inv2Rcut[nStubs];
  KFstateN::TR   cut_inv2R_minus = inv2RcutMinus[nStubs];
  KFstateN::TCHI cut_chi2        = chi2Cut[nStubs];
  // Don't do "hls::abs(x_new._3) <= cut_z0)" as this wastes 2 clk cycles.
  // Also, don't do "cut_z0_minus = - cut_z0" or this fails Vivado implementation with timing errors.
  selectOut.z0Cut = ((x_new._3 >= cut_z0_minus    && x_new._3 <= cut_z0)    || (cut_z0 == 0));  // cut = 0 means cut not applied.
  selectOut.ptCut = ((x_new._0 >= cut_inv2R_minus && x_new._0 <= cut_inv2R) || (cut_inv2R == 0)); 
  selectOut.chiSquaredCut = ((chi2_phi / chi2rphiScale + chi2_z <= cut_chi2) || (cut_chi2 == 0));
  selectOut.sufficientPScut = not (nStubs <= 2 && V._2Smodule);
  // IRT -- very useful whilst optimising variable bit ranges, to skip all but first iteration.
  //selectOut.ptCut = false;

  //=== Set output helix params & associated cov matrix related to d0, & check if d0 passes cut.
  //=== (Relevant only to 5-param helix fit) 
  setOutputsD0(x_new, C_new, nStubs, stateOut, selectOut);
  
#ifdef PRINT_HLSARGS
  stub.print("HLS INPUT stub:");
  stateIn.print("HLS INPUT state:");
  stateOut.print("HLS OUTPUT state:");
  selectOut.print("HLS OUTPUT extra:");
#endif
}

//=== Calculate increase in chi2 (in r-phi & r-z) from adding new stub: delta(chi2) = res(transpose) * R(inverse) * res

template <unsigned int NPAR>
void calcDeltaChi2(const VectorRes<NPAR>& res, const MatrixInverseR<NPAR>& Rinv, TCHI_INT& dChi2_phi, TCHI_INT& dChi2_z) {
  // Simplify calculation by noting that Rinv is symmetric.
  typedef typename MatrixInverseR<NPAR>::TRI00_short TRI00_short;
  typedef typename MatrixInverseR<NPAR>::TRI11_short TRI11_short;
  typedef typename MatrixInverseR<NPAR>::TRI01_short TRI01_short;
  dChi2_phi = (res._0 * res._0) * TRI00_short(Rinv._00) +
               2 * (res._0 * res._1) * TRI01_short(Rinv._01);
  dChi2_z   = (res._1 * res._1) * TRI11_short(Rinv._11);
#ifdef PRINT_SUMMARY
  double chi2_00 = double(res._0) * double(res._0) * double(Rinv._00);
  double chi2_01 = double(res._0) * double(res._1) * double(Rinv._01);
  double chi2_11 = double(res._1) * double(res._1) * double(Rinv._11);
  CHECK_AP::checkCalc("dChi2_phi", dChi2_phi, chi2_00 + 2*chi2_01, 0.1, 0.1);
  CHECK_AP::checkCalc("dChi2_z"  , dChi2_z  , chi2_11            , 0.1, 0.1);
#ifdef PRINT
  std::cout<<"Delta chi2_phi = "<<dChi2_phi<<" delta chi2_z = "<<dChi2_z<<" res (phi,z) = "<<res._0<<" "<<res._1<<std::endl;
#endif
#endif
  return;
}

//=== Set output helix params & associated cov matrix related to d0, & check if d0 passes cut.
//=== (Relevant only to 5-param helix fit)

void setOutputsD0(const VectorX<4>& x_new, const MatrixC<4>& C_new, const ap_uint<3>& nStubs, KFstate<4>& stateOut, KFselect<4>& selectOut) {}

void setOutputsD0(const VectorX<5>& x_new, const MatrixC<5>& C_new, const ap_uint<3>& nStubs, KFstate<5>& stateOut, KFselect<5>& selectOut) {
  stateOut.d0 = x_new._4;
  stateOut.cov_44 = C_new._44;
  stateOut.cov_04 = C_new._04;
  stateOut.cov_14 = C_new._14;
  KFstateN::TD cut_d0        = d0Cut[nStubs];
  KFstateN::TD cut_d0_minus  = d0CutMinus[nStubs];
  selectOut.d0Cut = ((x_new._4 >= cut_d0_minus && x_new._4 <= cut_d0) || (cut_d0 == 0));
}

  // ----- The following code is now done in VHDL at end of KF, so no longer needed in HLS. -----
  // ----- It used to be run at the end of kalmanUpdate(...)                                -----

  /*

  typename KFstateN::TP phiAtRefR = x_new._1 - chosenRofPhi * x_new._0;
  KFstubN::TZ             zAtRefR = x_new._3 + chosenRofZ * x_new._2; // Intentional use of KFstubN::TZ type

  // Constants BMH & BCH below set in KFconstants.h
  // Casting from ap_fixed to ap_int rounds to zero, not -ve infinity, so cast to ap_fixed with no fractional part first.
  ap_int<BMH> mBin_fit_tmp = ap_fixed<BMH,BMH>( 
					       ap_fixed<B18+inv2RToMbin_bitShift,KFstateN::BH0+inv2RToMbin_bitShift>(x_new._0) << inv2RToMbin_bitShift
					       );
  ap_int<BCH> cBin_fit_tmp = ap_fixed<BCH,BCH>(
					       ap_fixed<B18+phiToCbin_bitShift,KFstateN::BH1>(phiAtRefR) >> phiToCbin_bitShift
					       );
  bool cBinInRange = (cBin_fit_tmp >= minPhiBin && cBin_fit_tmp <= maxPhiBin);

  // Duplicate removal works best in mBin_fit is forced back into HT array if it lies just outside.
  KFstateN::TM mBin_fit_tmp_trunc;
  if (mBin_fit_tmp < minPtBin) {
    mBin_fit_tmp_trunc = minPtBin;
  } else if (mBin_fit_tmp > maxPtBin) {
    mBin_fit_tmp_trunc = maxPtBin;
  } else {
    mBin_fit_tmp_trunc = mBin_fit_tmp;
  }
  KFstateN::TC cBin_fit_tmp_trunc = cBin_fit_tmp;
  selectOut.mBin_fit = mBin_fit_tmp_trunc;
  selectOut.cBin_fit = cBin_fit_tmp_trunc;
  //std::cout<<"MBIN helix "<<selectOut.mBin_fit<<" tmp "<<mBin_fit_tmp<<" ht "<<stateIn.mBin_ht<<std::endl;
  //std::cout<<"CBIN helix "<<selectOut.cBin_fit<<" tmp "<<cBin_fit_tmp<<" ht "<<stateIn.cBin_ht<<std::endl;

  static const EtaBoundaries etaBounds;

  //for (unsigned int i = 0; i < 9+1; i++) std::cout<<"ETA "<<i<<" "<<int(etaBounds.z_[i])/2<<std::endl;

  // IRT -- feed in test helix params to debug cases seen in QuestaSim.
  //bool TMPSIGN = false;
  //if (TMPSIGN) zAtRefR = -zAtRefR;
  //unsigned int TMPS = 1;
  //bool inEtaSector = (zAtRefR > etaBounds.z_[TMPS] && zAtRefR < etaBounds.z_[TMPS+1]);

  if (stateIn.etaSectZsign == 1) zAtRefR = -zAtRefR;
  bool inEtaSector = (zAtRefR > etaBounds.z_[stateIn.etaSectID] && zAtRefR < etaBounds.z_[stateIn.etaSectID+1]);

  selectOut.sectorCut = (cBinInRange && inEtaSector);
  selectOut.consistent = (mBin_fit_tmp_trunc == stateIn.mBin_ht && cBin_fit_tmp_trunc == stateIn.cBin_ht);

  //std::cout<<"ZCALC "<<x_new._3<<" "<<chosenRofZ<<" "<<x_new._2<<std::endl;

  // IRT -- feed in test helix params to debug cases seen in QuestaSim.
  //std::cout<<"ZZZ RANGE TMP "<<etaBounds.z_[TMPS]<<" < "<<zAtRefR<<" < "<<etaBounds.z_[TMPS+1]<<" sec="<<TMPS<<" zsign="<<TMPSIGN<<std::endl;

  //std::cout<<"ZZZ RANGE "<<etaBounds.z_[stateIn.etaSectID]<<" < "<<zAtRefR<<" < "<<etaBounds.z_[stateIn.etaSectID+1]<<" sec="<<stateIn.etaSectID<<" zsign="<<stateIn.etaSectZsign<<std::endl;

  //std::cout<<"CHECK IN RANGE: c"<<cBinInRange<<" sec "<<inEtaSector<<std::endl;
  
  //std::cout<<"EXTRA: z0Cut="<<selectOut.z0Cut<<" ptCut="<<selectOut.ptCut<<" chi2Cut="<<selectOut.chiSquaredCut<<" PScut="<<selectOut.sufficientPScut<<std::endl;
  //std::cout<<"EXTRA: mBin="<<int(stateIn.mBin_ht)<<" "<<int(mBin_fit_tmp)<<" cBin="<<int(stateIn.cBin_ht)<<" "<<int(cBin_fit_tmp)<<" consistent="<<selectOut.consistent<<std::endl;
  //std::cout<<"EXTRA: in sector="<<selectOut.sectorCut<<" in eta="<<inEtaSector<<" phiAtR="<<phiAtRefR<<" zAtR="<<zAtRefR<<std::endl;

  */


#ifdef CMSSW_GIT_HASH
}

}
#endif
