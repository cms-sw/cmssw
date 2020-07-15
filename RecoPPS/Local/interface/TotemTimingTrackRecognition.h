/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoPPS_Local_TotemTimingTrackRecognition
#define RecoPPS_Local_TotemTimingTrackRecognition

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"

#include "RecoPPS/Local/interface/CTPPSTimingTrackRecognition.h"

/**
 * Class intended to perform general CTPPS timing detectors track recognition,
 * as well as construction of specialized classes (for now CTPPSDiamond and TotemTiming local tracks).
**/
class TotemTimingTrackRecognition : public CTPPSTimingTrackRecognition<TotemTimingLocalTrack, TotemTimingRecHit> {
public:
  TotemTimingTrackRecognition(const edm::ParameterSet& iConfig);

  // Adds new hit to the set from which the tracks are reconstructed.
  void addHit(const TotemTimingRecHit& recHit) override;

  /// Produces a collection of tracks for the current station, given its hits collection
  int produceTracks(edm::DetSet<TotemTimingLocalTrack>& tracks) override;
};

#endif
