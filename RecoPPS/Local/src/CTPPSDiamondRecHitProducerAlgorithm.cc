/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include <memory>
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoPPS/Local/interface/CTPPSDiamondRecHitProducerAlgorithm.h"

//----------------------------------------------------------------------------------------------------

CTPPSDiamondRecHitProducerAlgorithm::CTPPSDiamondRecHitProducerAlgorithm(const edm::ParameterSet& iConfig)
    : ts_to_ns_(iConfig.getParameter<double>("timeSliceNs")),
      apply_calib_(iConfig.getParameter<bool>("applyCalibration")) {}

void CTPPSDiamondRecHitProducerAlgorithm::setCalibration(const PPSTimingCalibration& calib,
                                                         const PPSTimingCalibrationLUT& calibLUT) {
  calib_ = calib;
  calibLUT_ = calibLUT;
  calib_fct_ = std::make_unique<reco::FormulaEvaluator>(calib_.formula());
}

void CTPPSDiamondRecHitProducerAlgorithm::build(const CTPPSGeometry& geom,
                                                const edm::DetSetVector<CTPPSDiamondDigi>& input,
                                                edm::DetSetVector<CTPPSDiamondRecHit>& output) {
  for (const auto& vec : input) {
    const CTPPSDiamondDetId detid(vec.detId());

    if (detid.channel() > MAX_CHANNEL)  // VFAT-like information, to be ignored
      continue;

    // retrieve the geometry element associated to this DetID
    const DetGeomDesc* det = geom.sensor(detid);

    const float x_pos = det->translation().x(), y_pos = det->translation().y();
    float z_pos = 0.;
    z_pos = det->parentZPosition();  // retrieve the plane position;

    // parameters stand for half the size
    const auto& diamondDimensions = det->getDiamondDimensions();
    const float x_width = 2.0 * diamondDimensions.xHalfWidth;
    const float y_width = 2.0 * diamondDimensions.yHalfWidth;
    const float z_width = 2.0 * diamondDimensions.zHalfWidth;

    const int sector = detid.arm(), station = detid.station(), plane = detid.plane(), channel = detid.channel();
    //LUT calibration
    std::vector<double> lut;
    if (apply_calib_) {
      lut = calibLUT_.bins(sector, station, plane, channel);
      if (lut.size() != 1024)
        lut = std::vector<double>(1024, 0.0);
    } else
      lut = std::vector<double>(1024, 0.0);

    // retrieve the timing calibration part for this channel
    const auto& ch_params = (apply_calib_) ? calib_.parameters(sector, station, plane, channel) : std::vector<double>{};
    // default values for offset + time precision if calibration object not found
    const double ch_t_offset = (apply_calib_) ? calib_.timeOffset(sector, station, plane, channel) : 0.;
    const double ch_t_precis = (apply_calib_) ? calib_.timePrecision(sector, station, plane, channel) : 0.;

    edm::DetSet<CTPPSDiamondRecHit>& rec_hits = output.find_or_insert(detid);

    for (const auto& digi : vec) {
      const int t_lead = digi.leadingEdge(), t_trail = digi.trailingEdge();
      // skip invalid digis
      if (t_lead == 0 && t_trail == 0)
        continue;

      double tot = -1., ch_t_twc = 0.;
      if (t_lead != 0 && t_trail != 0) {
        tot = (t_trail - t_lead) * ts_to_ns_;  // in ns
        if (calib_fct_ && apply_calib_ && !ch_params.empty()) {
          // compute the time-walk correction
          ch_t_twc = calib_fct_->evaluate(std::vector<double>{tot}, ch_params);
          if (edm::isNotFinite(ch_t_twc))
            ch_t_twc = 0.;
        }
      }

      const int time_slice =
          (t_lead != 0) ? (t_lead - ch_t_offset / ts_to_ns_) / 1024 : CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING;

      // calibrated time of arrival
      const double t0 = (t_lead % 1024) * ts_to_ns_ + lut[t_lead % 1024] - ch_t_twc;
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
          digi.hptdcErrorFlags(),
          digi.multipleHit());
    }
  }
}
