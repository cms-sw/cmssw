/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingRecHitProducerAlgorithm.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------------------------------------------------------------------------------------------

const float TotemTimingRecHitProducerAlgorithm::SINC_COEFFICIENT = std::acos(-1) * 2 / 7.8;

//----------------------------------------------------------------------------------------------------

TotemTimingRecHitProducerAlgorithm::TotemTimingRecHitProducerAlgorithm(
    const edm::ParameterSet &iConfig)
    : sampicConversions_(iConfig.getParameter<std::string>("calibrationFile")),
      baselinePoints_(iConfig.getParameter<int>("baselinePoints")),
      saturationLimit_(iConfig.getParameter<double>("saturationLimit")),
      cfdFraction_(iConfig.getParameter<double>("cfdFraction")),
      smoothingPoints_(iConfig.getParameter<int>("smoothingPoints")),
      lowPassFrequency_(iConfig.getParameter<double>("lowPassFrequency")),
      hysteresis_(iConfig.getParameter<double>("hysteresis")) {}

void TotemTimingRecHitProducerAlgorithm::build(
    const CTPPSGeometry *geom, const edm::DetSetVector<TotemTimingDigi> &input,
    edm::DetSetVector<TotemTimingRecHit> &output) {
  for (const auto &vec : input) {
    const TotemTimingDetId detid(vec.detId());

    float x_pos = 0.0f, y_pos = 0.0f, z_pos = 0.0f, x_width = 0.0f, y_width = 0.0f, 
          z_width = 0.0f;

    // retrieve the geometry element associated to this DetID ( if present )
    const DetGeomDesc *det = geom->getSensorNoThrow(detid);

    if (det) {
      x_pos = det->translation().x(), y_pos = det->translation().y();
      z_pos = det->parentZPosition(); // retrieve the plane position;

      x_width = 2.0 * det->params()[0], // parameters stand for half the size
      y_width = 2.0 * det->params()[1],
      z_width = 2.0 * det->params()[2];
    } else
      edm::LogWarning("TotemTimingRecHitProducerAlgorithm")
          << "Failed to retrieve a sensor for " << detid;

    edm::DetSet<TotemTimingRecHit> &rec_hits = output.find_or_insert(detid);

    for (const auto &digi : vec) {
      const float triggerCellTimeInstant(
          sampicConversions_.getTriggerTime(digi));
      const std::vector<float> time(sampicConversions_.getTimeSamples(digi));
      std::vector<float> data(sampicConversions_.getVoltSamples(digi));

      auto max_it = std::max_element(data.begin(), data.end());

      RegressionResults baselineRegression =
          simplifiedLinearRegression(time, data, 0, baselinePoints_);

      // remove baseline
      std::vector<float> dataCorrected(data.size());
      for (unsigned int i = 0; i < data.size(); ++i)
        dataCorrected[i] = data[i] -
                   (baselineRegression.q + baselineRegression.m * time[i]);
      auto max_corrected_it =
          std::max_element(dataCorrected.begin(), dataCorrected.end());

      float t = TotemTimingRecHit::NO_T_AVAILABLE;
      if (*max_it < saturationLimit_)
        t = constantFractionDiscriminator(time, dataCorrected);

      mode_ = TotemTimingRecHit::CFD;

      rec_hits.push_back(TotemTimingRecHit(
          x_pos, x_width, y_pos, y_width, z_pos, z_width, // spatial information
          t, triggerCellTimeInstant, .0, *max_corrected_it,
          baselineRegression.rms, mode_));
    }
  }
}

TotemTimingRecHitProducerAlgorithm::RegressionResults
TotemTimingRecHitProducerAlgorithm::simplifiedLinearRegression(
    const std::vector<float> &time, const std::vector<float> &data,
    const unsigned int start_at, const unsigned int points) const {
  RegressionResults results;
  if (time.size() != data.size()) {
    return results;
  }
  if (start_at > time.size()) {
    return results;
  }
  unsigned int stop_at = std::min((unsigned int)time.size(), start_at + points);
  unsigned int realPoints = stop_at - start_at;
  auto t_begin = std::next(time.begin(), start_at);
  auto t_end = std::next(time.begin(), stop_at);
  auto d_begin = std::next(data.begin(), start_at);
  auto d_end = std::next(data.begin(), stop_at);

  float sx = .0f;
  std::for_each(t_begin, t_end, [&](float value) { sx += value; });
  float sxx = .0f;
  std::for_each(t_begin, t_end, [&](float value) { sxx += value * value; });

  float sy = .0f;
  std::for_each(d_begin, d_end, [&](float value) { sy += value; });
  float syy = .0f;
  std::for_each(d_begin, d_end, [&](float value) { syy += value * value; });

  float sxy = .0f;
  for (unsigned int i = 0; i < realPoints; ++i)
    sxy += (time[i]) * (data[i]);

  // y = mx + q
  results.m = (sxy * realPoints - sx * sy) / (sxx * realPoints - sx * sx);
  results.q = sy / realPoints - results.m * sx / realPoints;

  float correctedSyy = .0f;
  for (unsigned int i = 0; i < realPoints; ++i)
    correctedSyy += pow(data[i] - (results.m * time[i] + results.q), 2);
  results.rms = sqrt(correctedSyy / realPoints);

  return results;
}

int TotemTimingRecHitProducerAlgorithm::fastDiscriminator(
    const std::vector<float> &data, const float &threshold) const {
  int threholdCrossingIndex = -1;
  bool above = false;
  bool lockForHysteresis = false;

  for (unsigned int i = 0; i < data.size(); ++i) {
    // Look for first edge
    if (!above && !lockForHysteresis && data[i] > threshold) {
      threholdCrossingIndex = i;
      above = true;
      lockForHysteresis = true;
    }
    if (above && lockForHysteresis) // NOTE: not else if!, "above" can be set in
                                    // the previous if
    {
      // Lock until above threshold_+hysteresis
      if (lockForHysteresis && data[i] > threshold + hysteresis_) {
        lockForHysteresis = false;
      }
      // Ignore noise peaks
      if (lockForHysteresis && data[i] < threshold) {
        above = false;
        lockForHysteresis = false;
        threholdCrossingIndex = -1; // assigned because of noise
      }
    }
  }

  return threholdCrossingIndex;
}

float TotemTimingRecHitProducerAlgorithm::constantFractionDiscriminator(
    const std::vector<float> &time, const std::vector<float> &data) {
  std::vector<float> dataProcessed(data);
  if (lowPassFrequency_ != 0) {
    // Smoothing
    for (int i = 0; i < (int)data.size(); ++i) {
      for (int j = -smoothingPoints_ / 2;
           j <= +smoothingPoints_ / 2; ++j) {
        if ((i + j) >= 0 && (i + j) < (int)data.size() && j != 0) {
          float x = SINC_COEFFICIENT * lowPassFrequency_ * j;
          dataProcessed[i] += data[i + j] * std::sin(x) / x;
        }
      }
    }
  }
  auto max_it = std::max_element(dataProcessed.begin(), dataProcessed.end());
  float max = *max_it;

  float threshold = cfdFraction_ * max;
  int indexOfThresholdCrossing = fastDiscriminator(dataProcessed, threshold);

  float t = TotemTimingRecHit::NO_T_AVAILABLE;
  if (indexOfThresholdCrossing >= baselinePoints_ &&
      indexOfThresholdCrossing < (int)time.size()) {
    t = (time[indexOfThresholdCrossing - 1] -
         time[indexOfThresholdCrossing]) /
            (dataProcessed[indexOfThresholdCrossing - 1] -
             dataProcessed[indexOfThresholdCrossing]) *
            (threshold - dataProcessed[indexOfThresholdCrossing]) +
        time[indexOfThresholdCrossing];
  }

  return t;
}
