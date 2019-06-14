/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondRecHitProducerAlgorithm.h"
#include "FWCore/Utilities/interface/isFinite.h"

//----------------------------------------------------------------------------------------------------

CTPPSDiamondRecHitProducerAlgorithm::CTPPSDiamondRecHitProducerAlgorithm(const edm::ParameterSet& iConfig)
    : ts_to_ns_(iConfig.getParameter<double>("timeSliceNs")) {}

void CTPPSDiamondRecHitProducerAlgorithm::setCalibration(const PPSTimingCalibration& calib) {
  calib_ = calib;
  calib_fct_.reset(new reco::FormulaEvaluator(calib_.formula()));
}

void CTPPSDiamondRecHitProducerAlgorithm::build(const CTPPSGeometry& geom,
                                                const edm::DetSetVector<CTPPSDiamondDigi>& input,
                                                edm::DetSetVector<CTPPSDiamondRecHit>& output) {
  for (const auto& vec : input) {
    const CTPPSDiamondDetId detid(vec.detId());

    if (detid.channel() > MAX_CHANNEL)  // VFAT-like information, to be ignored
      continue;

    // retrieve the geometry element associated to this DetID
    const DetGeomDesc* det = geom.getSensor(detid);

    const float x_pos = det->translation().x(), y_pos = det->translation().y();
    float z_pos = 0.;
    z_pos = det->parentZPosition();  // retrieve the plane position;

    const float x_width = 2.0 * det->params().at(0),  // parameters stand for half the size
        y_width = 2.0 * det->params().at(1), z_width = 2.0 * det->params().at(2);

    // retrieve the timing calibration part for this channel
    const int sector = detid.arm(), station = detid.station(), plane = detid.plane(), channel = detid.channel();
    const auto& ch_params = calib_.parameters(sector, station, plane, channel);
    // offset + time precision set to 0 if not found
    const double ch_t_offset = calib_.timeOffset(sector, station, plane, channel);
    const double ch_t_precis = calib_.timePrecision(sector, station, plane, channel);

    edm::DetSet<CTPPSDiamondRecHit>& rec_hits = output.find_or_insert(detid);

    for (const auto& digi : vec) {
      const int t_lead = digi.getLeadingEdge(), t_trail = digi.getTrailingEdge();
      // skip invalid digis
      if (t_lead == 0 && t_trail == 0)
        continue;

      double tot = -1., ch_t_twc = 0.;
      if (t_lead != 0 && t_trail != 0) {
        tot = (t_trail - t_lead) * ts_to_ns_;  // in ns
        if (calib_fct_) {
          // compute the time-walk correction
          ch_t_twc = calib_fct_->evaluate(std::vector<double>{tot}, ch_params);
          if (edm::isNotFinite(ch_t_twc))
            ch_t_twc = 0.;
        }
      }

      const int time_slice =
          (t_lead != 0) ? (t_lead - ch_t_offset / ts_to_ns_) / 1024 : CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING;

      // calibrated time of arrival
      const double t0 = (t_lead % 1024) * ts_to_ns_ - ch_t_twc;

      rec_hits.emplace_back(
          // spatial information
          x_pos,
          x_width,
          y_pos,
          y_width,
          z_pos,
          z_width,
          // timing information
          t0,
          tot,
          ch_t_precis,
          time_slice,
          // readout information
          digi.getHPTDCErrorFlags(),
          digi.getMultipleHit());
    }
  }
}
