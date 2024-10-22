/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoPPS_Local_CTPPSDiamondTrackRecognition
#define RecoPPS_Local_CTPPSDiamondTrackRecognition

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "RecoPPS/Local/interface/CTPPSTimingTrackRecognition.h"

#include <unordered_map>

/**
 * \brief Class performing smart reconstruction for PPS Diamond Detectors.
 * \date Jan 2017
**/
class CTPPSDiamondTrackRecognition : public CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit> {
public:
  CTPPSDiamondTrackRecognition(const edm::ParameterSet& iConfig);

  void clear() override;
  /// Feed a new hit to the tracks recognition algorithm
  void addHit(const CTPPSDiamondRecHit& recHit) override;
  /// Produce a collection of tracks for the current station, given its hits collection
  int produceTracks(edm::DetSet<CTPPSDiamondLocalTrack>& tracks) override;

private:
  std::unordered_map<int, int> mhMap_;
  bool excludeSingleEdgeHits_;
};

#endif
