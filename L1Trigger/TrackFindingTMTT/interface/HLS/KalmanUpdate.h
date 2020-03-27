/**
 * This is the top-level HLS function within CMSSW, which updates a helix state by adding a stub to it.
 * N.B. It therefore can't use the Settings class or any external libraries! Nor can it be a C++ class.
 *
 * All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 * 
 * Author: Ian Tomalin
 */

 
#ifndef __KalmanUpdate__
#define __KalmanUpdate__

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstate.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices4.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanMatrices5.h"
#else
#include "HLSutilities.h"
#include "KFstub.h"
#include "KFstate.h"
#include "KalmanMatrices.h"
#include "KalmanMatrices4.h"
#include "KalmanMatrices5.h"
#endif
 
#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

// Internal interface.
// Add stub to old KF helix state to get new KF helix state for NPAR = 4 or 5 param helix fits.
template <unsigned int NPAR>
void kalmanUpdate(const KFstubC& stub, const KFstate<NPAR>& stateIn, KFstate<NPAR>& stateOut, KFselect<NPAR>& selectOut);

// Calculate increase in chi2 (in r-phi & r-z) from adding new stub: delta(chi2) = res(transpose) * R(inverse) * res
template <unsigned int NPAR>
void calcDeltaChi2(const VectorRes<NPAR>& res, const MatrixInverseR<NPAR>& Rinv, TCHI_INT& dChi2_phi, TCHI_INT& dChi2_z);

// Set output helix params & associated cov matrix related to d0, & check if d0 passes cut.
// (Relevant only to 5-param helix fit)
template <unsigned int NPAR>
void setOutputsD0(const VectorX<NPAR>& x_new, const MatrixC<NPAR>& C_new, const ap_uint<3>& nStubs, KFstate<NPAR>& stateOut, KFselect<NPAR>& selectOut);

// Fully specialized function templates must also be declared to ensure they are found.

#ifdef CMSSW_GIT_HASH
}

}
#endif

#endif




