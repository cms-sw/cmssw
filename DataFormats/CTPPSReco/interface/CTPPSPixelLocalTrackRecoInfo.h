/*
*
* This is a part of CTPPS offline software.
* Author:
*   Andrea Bellora (andrea.bellora@cern.ch)
*
*/
#ifndef DataFormats_CTPPSReco_CTPPSPixelLocalTrackRecoInfo_H
#define DataFormats_CTPPSReco_CTPPSPixelLocalTrackRecoInfo_H

/// Track information byte for bx-shifted runs:
/// reco_info = notShiftedRun    -> Default value for tracks reconstructed in non-bx-shifted ROCs
/// reco_info = allShiftedPlanes -> Track reconstructed in a bx-shifted ROC with bx-shifted planes only
/// reco_info = noShiftedPlanes  -> Track reconstructed in a bx-shifted ROC with non-bx-shifted planes only
/// reco_info = mixedPlanes      -> Track reconstructed in a bx-shifted ROC both with bx-shifted and non-bx-shifted planes
/// reco_info = invalid          -> Dummy value. Assigned when reco_info is not computed
enum class CTPPSpixelLocalTrackReconstructionInfo {
  notShiftedRun = 0,
  allShiftedPlanes = 1,
  noShiftedPlanes = 2,
  mixedPlanes = 3,
  invalid = 5
};

#endif
