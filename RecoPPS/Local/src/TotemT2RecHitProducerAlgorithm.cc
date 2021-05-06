/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "RecoPPS/Local/interface/TotemT2RecHitProducerAlgorithm.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"

void TotemT2RecHitProducerAlgorithm::build(const TotemGeometry& geom,
                                           const edm::DetSetVector<TotemT2Digi>& input,
                                           edm::DetSetVector<TotemT2RecHit>& output) {
  for (const auto& vec : input) {
    const TotemT2DetId detid(vec.detId());
    const int sector = detid.arm(), plane = detid.plane(), channel = detid.channel();

    // retrieve the timing calibration part for this channel
    const auto& ch_params = (apply_calib_) ? calib_.parameters(sector, 0, plane, channel) : std::vector<double>{};
    // default values for offset + time precision if calibration object not found
    const double ch_t_offset = (apply_calib_) ? calib_.timeOffset(sector, 0, plane, channel) : 0.;
    const double ch_t_precis = (apply_calib_) ? calib_.timePrecision(sector, 0, plane, channel) : 0.;

    // prepare the output collection
    edm::DetSet<TotemT2RecHit>& rec_hits = output.find_or_insert(detid);

    // retrieve the geometry element associated to this DetID
    const auto& tile = geom.tile(detid);

    rec_hits.emplace_back(tile.centre(), ch_t_offset, ch_t_precis);
  }
}
