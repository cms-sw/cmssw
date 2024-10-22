/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "RecoPPS/Local/interface/TotemT2RecHitProducerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"

void TotemT2RecHitProducerAlgorithm::build(const TotemGeometry& geom,
                                           const edmNew::DetSetVector<TotemT2Digi>& input,
                                           edmNew::DetSetVector<TotemT2RecHit>& output) {
  for (const auto& vec : input) {
    const TotemT2DetId detid(vec.detId());

    // prepare the output collection filler
    edmNew::DetSetVector<TotemT2RecHit>::FastFiller filler(output, detid);

    for (const auto& digi : vec) {
      const int t_lead = digi.leadingEdge(), t_trail = digi.trailingEdge();
      // don't skip no-edge digis
      double tot = 0.;
      if (digi.hasLE() && digi.hasTE()) {
        tot = (t_trail - t_lead) * ts_to_ns_;  // in ns
      }
      double ch_t_precis = ts_to_ns_ / 2.0;  //without a calibration, assume LE/TE precision is +-0.5

      // retrieve the geometry element associated to this DetID
      const auto& tile = geom.tile(detid);

      // store to the output collection
      filler.emplace_back(tile.centre(), t_lead * ts_to_ns_, ch_t_precis, tot);
    }
  }
}
