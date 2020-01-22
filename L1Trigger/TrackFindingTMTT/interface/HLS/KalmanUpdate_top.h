#ifndef __KalmanUpdate_top__
#define __KalmanUpdate_top__

/**
 * This is the top-level function for Vivado HLS compilation. 
 * It is not used by CMSSW.
 * 
 * It is required because HLS does not allow the top-level function to be templated.
 * 
 * Author: Ian Tomalin
 */

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstate.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanUpdate.h"
#else
#include "KFstub.h"
#include "KFstate.h"
#include "KalmanUpdate.h"
#endif

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

void kalmanUpdate_top(const KFstubC& stub, const KFstate<N_HELIX_PAR>& stateIn, KFstate<N_HELIX_PAR>& stateOut, KFselect<N_HELIX_PAR>& selectOut);

#ifdef CMSSW_GIT_HASH
}

}
#endif

#endif
