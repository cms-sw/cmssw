/**
 * This is the top-level function for Vivado HLS compilation. 
 * It is not used by CMSSW.
 * 
 * It is required because HLS does not allow the top-level function to be templated.
 * 
 * Author: Ian Tomalin
 */

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KalmanUpdate_top.h"
#else
#include "KalmanUpdate_top.h"
#endif
 
#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

void kalmanUpdate_top(const KFstubC& stub, const KFstate<N_HELIX_PAR>& stateIn, KFstate<N_HELIX_PAR>& stateOut, KFselect<N_HELIX_PAR>& selectOut) {

#pragma HLS PIPELINE II=1
  //#pragma HLS INTERFACE ap_ctrl_hs register port=return
  //#pragma HLS INTERFACE ap_none port=stub       register 
  //#pragma HLS INTERFACE ap_none port=stateIn    register 
  //#pragma HLS INTERFACE ap_none port=stateOut   register 
  //#pragma HLS INTERFACE ap_none port=selectOut  register 

#pragma HLS INTERFACE ap_ctrl_hs  port=return
#pragma HLS INTERFACE ap_none port=stub        
#pragma HLS INTERFACE ap_none port=stateIn     
#pragma HLS INTERFACE ap_none port=stateOut    
#pragma HLS INTERFACE ap_none port=selectOut   

  kalmanUpdate<N_HELIX_PAR>(stub, stateIn, stateOut, selectOut);
}

#ifdef CMSSW_GIT_HASH
}

}
#endif
