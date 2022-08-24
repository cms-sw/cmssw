/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoPPS/Local/interface/TotemT2RecHitProducerAlgorithm.h"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"

void TotemT2RecHitProducerAlgorithm::build(const TotemGeometry& geom,
                                           const edmNew::DetSetVector<TotemT2Digi>& input,
                                           edmNew::DetSetVector<TotemT2RecHit>& output) {
  for (const auto& vec : input) {
    const TotemT2DetId detid(vec.detId());
    const int sector = detid.arm(), plane = detid.plane(), channel = detid.channel();

    // retrieve the timing calibration part for this channel
    const auto& ch_params = (apply_calib_) ? calib_->parameters(sector, 0, plane, channel) : std::vector<double>{};
    // default values for offset + time precision if calibration object not found
    const double ch_t_offset = (apply_calib_) ? calib_->timeOffset(sector, 0, plane, channel) : 0.;
    const double ch_t_precis = (apply_calib_) ? calib_->timePrecision(sector, 0, plane, channel) : 0.;

    // prepare the output collection filler
    edmNew::DetSetVector<TotemT2RecHit>::FastFiller filler(output, detid);

    for (const auto& digi : vec) {
      const int t_lead = digi.leadingEdge(), t_trail = digi.trailingEdge();
      if (t_lead == 0 && t_trail == 0)  // skip invalid digis
        continue;
      double tot = -1., ch_t_twc = 0.;
      if (t_lead != 0 && t_trail != 0) {
        tot = (t_trail - t_lead) * ts_to_ns_;  // in ns
        if (calib_fct_ && apply_calib_) {      // compute the time-walk correction
          ch_t_twc = calib_fct_->evaluate(std::vector<double>{tot}, ch_params);
          if (edm::isNotFinite(ch_t_twc))
            ch_t_twc = 0.;
        }
      }

      // retrieve the geometry element associated to this DetID
      const auto& tile = geom.tile(detid);

      // store to the output collection
      filler.emplace_back(tile.centre(), t_lead * ts_to_ns_ - ch_t_offset - ch_t_twc, ch_t_precis, tot);
    }
  }
}
