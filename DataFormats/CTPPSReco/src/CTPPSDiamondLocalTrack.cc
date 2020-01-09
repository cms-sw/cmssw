/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

//--- constructors

CTPPSDiamondLocalTrack::CTPPSDiamondLocalTrack() : ts_index_(0), mh_(0) {}

CTPPSDiamondLocalTrack::CTPPSDiamondLocalTrack(
    const math::XYZPoint& pos0, const math::XYZPoint& pos0_sigma, float t, float t_sigma, int oot_idx, int mult_hits)
    : CTPPSTimingLocalTrack(pos0, pos0_sigma, t, t_sigma), ts_index_(oot_idx), mh_(mult_hits) {}

//--- interface member functions

bool CTPPSDiamondLocalTrack::containsHit(const CTPPSDiamondRecHit& recHit, float tolerance) const {
  if (!CTPPSTimingLocalTrack::containsHit(recHit, tolerance))
    return false;

  return (recHit.ootIndex() == ts_index_ ||
          recHit.ootIndex() == ts_index_ + CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING);
}
