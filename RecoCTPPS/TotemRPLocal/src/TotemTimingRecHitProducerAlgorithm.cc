/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingRecHitProducerAlgorithm.h"

//----------------------------------------------------------------------------------------------------

const float TotemTimingRecHitProducerAlgorithm::NO_T_AVAILABLE = -100;

//----------------------------------------------------------------------------------------------------

TotemTimingRecHitProducerAlgorithm::TotemTimingRecHitProducerAlgorithm(
    const edm::ParameterSet &iConfig)
    : sampicConversions_(iConfig.getParameter<std::string>("calibrationFile")),
      baselinePoints_(iConfig.getParameter<int>("baselinePoints")),
      risingEdgePointsBeforeTh_(iConfig.getParameter<int>("risingEdgePointsBeforeTh")),
      risingEdgePoints_(iConfig.getParameter<int>("risingEdgePoints")),
      saturationLimit_(iConfig.getParameter<double>("saturationLimit")),
      thresholdFactor_(iConfig.getParameter<double>("thresholdFactor")),
      cfdFraction_(iConfig.getParameter<double>("cfdFraction")),
      hysteresis_(iConfig.getParameter<double>("hysteresis")) {}

void TotemTimingRecHitProducerAlgorithm::build(
    const CTPPSGeometry *geom, const edm::DetSetVector<TotemTimingDigi> &input,
    edm::DetSetVector<TotemTimingRecHit> &output) {
  for (const auto &vec : input) {
    const TotemTimingDetId detid(vec.detId());

    float x_pos = 0, y_pos = 0, z_pos = 0, x_width = 0, y_width = 0,
          z_width = 0;

    // retrieve the geometry element associated to this DetID ( if present )
    const DetGeomDesc *det = nullptr;
    try { // no other efficient way to check presence
      det = geom->getSensor(detid);
    } catch (cms::Exception &) {
      det = nullptr;
    }

    if (det != nullptr) {
      x_pos = det->translation().x(), y_pos = det->translation().y();
      if (det->parents().empty())
        edm::LogWarning("TotemTimingRecHitProducerAlgorithm")
            << "The geometry element for " << detid
            << " has no parents. Check the geometry hierarchy!";
      else
        z_pos = det->parents()[det->parents().size() - 1]
                    .absTranslation()
                    .z(); // retrieve the plane position;

      x_width = 2.0 * det->params().at(0), // parameters stand for half the size
          y_width = 2.0 * det->params().at(1),
      z_width = 2.0 * det->params().at(2);
    }

    edm::DetSet<TotemTimingRecHit> &rec_hits = output.find_or_insert(detid);

    for (const auto &digi : vec) {
      const float triggerCellTimeInstant(sampicConversions_.getTriggerTime(digi));
      const std::vector<float> time(sampicConversions_.getTimeSamples(digi));
      std::vector<float> data(sampicConversions_.getVoltSamples(digi));

      auto max_it = std::max_element(data.begin(), data.end());

      float t = NO_T_AVAILABLE;

      RegressionResults baselineRegression =
          simplifiedLinearRegression(time, data, 0, baselinePoints_);

      // remove baseline
      std::vector<float> dataCorrected(data.size());
      for (unsigned int i = 0; i < data.size(); ++i)
        dataCorrected.at(i) = data.at(i) - (baselineRegression.q);
      auto max_corrected_it =
          std::max_element(dataCorrected.begin(), dataCorrected.end());

      if (*max_it < saturationLimit_)
        t = constantFractionDiscriminator(time, dataCorrected);
      if (t!=NO_T_AVAILABLE)
        t -= triggerCellTimeInstant;

      float t_smart = smartTimeOfArrival(time, dataCorrected,
                             thresholdFactor_ * baselineRegression.rms);
      if (t_smart!=NO_T_AVAILABLE)
        t_smart += triggerCellTimeInstant;

      mode_ = TotemTimingRecHit::CFD;

      rec_hits.push_back(TotemTimingRecHit(
          x_pos, x_width, y_pos, y_width, z_pos, z_width, // spatial information
          t, triggerCellTimeInstant, t_smart, *max_corrected_it,
          baselineRegression.rms, mode_));

      std::cout<<"rechit: triggert: "<<triggerCellTimeInstant<<"\tfirstcell: "<<time.at(0)<< "\tt CFD: "<< t<<"\tt smart: "<<t_smart<<"\tmax corrected: "<<*max_it<<"\n";
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
  if (start_at > data.size()) {
    return results;
  }
  unsigned int stop_at = std::min((unsigned int)time.size(), start_at + points);
  unsigned int realPoints = stop_at - start_at;
  auto t_begin = std::next(time.begin(), start_at);
  auto t_end = std::next(time.begin(), stop_at);
  auto d_begin = std::next(data.begin(), start_at);
  auto d_end = std::next(data.begin(), stop_at);

  float sx = .0;
  std::for_each(t_begin, t_end, [&](float value) { sx += value; });
  float sxx = .0;
  std::for_each(t_begin, t_end, [&](float value) { sxx += value * value; });

  float sy = .0;
  std::for_each(d_begin, d_end, [&](float value) { sy += value; });
  float syy = .0;
  std::for_each(d_begin, d_end, [&](float value) { syy += value * value; });

  float sxy = .0;
  for (unsigned int i = 0; i < realPoints; ++i)
    sxy += (time.at(i)) * (data.at(i));

  // y = mx + q
  results.m = (sxy * realPoints - sx * sy) / (sxx * realPoints - sx * sx);
  results.q = sy / realPoints - results.m * sx / realPoints;

  float correctedSyy = .0;
  for (unsigned int i = 0; i < realPoints; ++i)
    correctedSyy += pow(data.at(i) - (results.m * time.at(i) + results.q), 2);
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
    if (!above && !lockForHysteresis && data.at(i) > threshold) {
      threholdCrossingIndex = i;
      above = true;
      lockForHysteresis = true;
    }
    if (above && lockForHysteresis) // NOTE: not else if!, "above" can be set in
                                    // the previous if
    {
      // Lock until above threshold_+hysteresis
      if (lockForHysteresis && data.at(i) > threshold + hysteresis_) {
        lockForHysteresis = false;
      }
      // Ignore noise peaks
      if (lockForHysteresis && data.at(i) < threshold) {
        above = false;
        lockForHysteresis = false;
        threholdCrossingIndex = -1; // assigned because of noise
      }
    }
  }

  return threholdCrossingIndex;
}

float TotemTimingRecHitProducerAlgorithm::smartTimeOfArrival(
    const std::vector<float> &time, const std::vector<float> &data,
    const float threshold) {
  int indexOfThresholdCrossing = fastDiscriminator(data, threshold);

  float t = NO_T_AVAILABLE;
  if (indexOfThresholdCrossing - risingEdgePointsBeforeTh_ > 0) {
    RegressionResults risingEdgeRegression = simplifiedLinearRegression(
        time, data, indexOfThresholdCrossing - risingEdgePointsBeforeTh_,
        risingEdgePoints_);
    // Find intersection with zero (baseline subtracted before)
    if (risingEdgeRegression.m > 0.1 && risingEdgeRegression.m < 5)
      t = (.0 - risingEdgeRegression.q / risingEdgeRegression.m);
  }

  return t;
}

float TotemTimingRecHitProducerAlgorithm::constantFractionDiscriminator(
    const std::vector<float> &time, const std::vector<float> &data) {
  auto max_it = std::max_element(data.begin(), data.end());
  float max = *max_it;

  float threshold = cfdFraction_ * max;
  int indexOfThresholdCrossing = fastDiscriminator(data, threshold);

  float t = NO_T_AVAILABLE;
  if (indexOfThresholdCrossing>=baselinePoints_) {
    RegressionResults risingEdgeRegression = simplifiedLinearRegression(
        time, data, indexOfThresholdCrossing - risingEdgePoints_/2,
        risingEdgePoints_);
    // Find intersection with threshold
      t = (threshold - risingEdgeRegression.q / risingEdgeRegression.m);
  }
  // if (indexOfThresholdCrossing>=baselinePoints_ && indexOfThresholdCrossing<(int)time.size())
  // {
  //   t = (time.at(indexOfThresholdCrossing-1) - time.at(indexOfThresholdCrossing)) /
  //       (data.at(indexOfThresholdCrossing-1) - data.at(indexOfThresholdCrossing)) *
  //       (threshold - data.at(indexOfThresholdCrossing)) + time.at(indexOfThresholdCrossing);
  // }
  std::cout<<"rechit: index: "<<indexOfThresholdCrossing<<"\ttime"<<t<<"\n";

  return t;
}
