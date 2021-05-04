/****************************************************************************
 *
 * F.Ferro ferro@ge.infn.it
 *
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"

void CTPPSPixelAnalysisMask::insert(const uint32_t &sid, const CTPPSPixelROCAnalysisMask &am) {
  analysisMask[sid] = am;
}
