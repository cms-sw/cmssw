/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSDiamondLocalTrack
#define DataFormats_CTPPSReco_CTPPSDiamondLocalTrack

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

//----------------------------------------------------------------------------------------------------

class CTPPSDiamondLocalTrack : public CTPPSTimingLocalTrack {
public:
  CTPPSDiamondLocalTrack();
  CTPPSDiamondLocalTrack(
      const math::XYZPoint& pos0, const math::XYZPoint& pos0_sigma, float t, float t_sigma, int oot_idx, int mult_hits);

  bool containsHit(const CTPPSDiamondRecHit& recHit, float tolerance = 0.1) const;

  //--- temporal set'ters

  inline void setOOTIndex(int i) { ts_index_ = i; }
  inline int ootIndex() const { return ts_index_; }

  inline void setMultipleHits(int i) { mh_ = i; }
  inline int multipleHits() const { return mh_; }

private:
  /// Time slice index
  int ts_index_;
  /// Multiple hits counter
  int mh_;
};

#endif
