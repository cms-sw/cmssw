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

const double TotemTimingRecHitProducerAlgorithm::SAMPIC_SAMPLING_PERIOD_NS =
    1. / 7.8;
const double TotemTimingRecHitProducerAlgorithm::SAMPIC_MAX_NUMBER_OF_SAMPLES =
    64;
const double TotemTimingRecHitProducerAlgorithm::SAMPIC_ADC_V = 1. / 256;

TotemTimingRecHitProducerAlgorithm::TotemTimingRecHitProducerAlgorithm(
    const edm::ParameterSet &iConfig)
    : baselinePoints_(iConfig.getParameter<unsigned int>("baselinePoints")),
      risingEdgePointsBeforeTh_(
          iConfig.getParameter<unsigned int>("risingEdgePointsBeforeTh")),
      risingEdgePoints_(iConfig.getParameter<unsigned int>("risingEdgePoints")),
      threholdFactor_(iConfig.getParameter<double>("threholdFactor")),
      cfdFraction_(iConfig.getParameter<double>("cfdFraction")),
      hysteresis_(iConfig.getParameter<double>("hysteresis")) {}

void TotemTimingRecHitProducerAlgorithm::build(
    const CTPPSGeometry *geom, const edm::DetSetVector<TotemTimingDigi> &input,
    edm::DetSetVector<TotemTimingRecHit> &output) {
  for (const auto &vec : input) {
    const TotemTimingDetId detid(vec.detId());

    /*
    // retrieve the geometry element associated to this DetID
    const DetGeomDesc* det = geom->getSensor( detid );

    const float x_pos = det->translation().x(),
                y_pos = det->translation().y();
    float z_pos = 0.;
    if ( det->parents().empty() )
      edm::LogWarning("TotemTimingRecHitProducerAlgorithm") << "The geometry
    element for " << detid << " has no parents. Check the geometry hierarchy!";
    else
      z_pos = det->parents()[det->parents().size()-1].absTranslation().z(); //
    retrieve the plane position;

    const float x_width = 2.0 * det->params().at( 0 ), // parameters stand for
    half the size y_width = 2.0 * det->params().at( 1 ), z_width = 2.0 *
    det->params().at( 2 );

                */
    const float x_pos = 0, y_pos = 0, z_pos = 0, x_width = 0, y_width = 0,
                z_width = 0;

    edm::DetSet<TotemTimingRecHit> &rec_hits = output.find_or_insert(detid);

    for (const auto &digi : vec) {

      // Time of samples
      unsigned int OFFSET_SAMPIC = digi.getEventInfo().getOffsetOfSamples();
      unsigned int cell0TimeClock;
      float cell0TimeInstant;       // Time of first cell
      float triggerCellTimeInstant; // Time of triggered cell

      unsigned int timestamp = digi.getCellInfo() <= 32 ? digi.getTimestampA()
                                                        : digi.getTimestampB();
      cell0TimeClock = timestamp +
                       ((digi.getFPGATimestamp() - timestamp) & 0xFFFFFFF000) -
                       digi.getEventInfo().getL1ATimestamp() +
                       digi.getEventInfo().getL1ALatency();
      cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES *
                         SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

      if (digi.getCellInfo() < OFFSET_SAMPIC)
        triggerCellTimeInstant =
            cell0TimeInstant + digi.getCellInfo() * SAMPIC_SAMPLING_PERIOD_NS;
      else
        triggerCellTimeInstant =
            cell0TimeInstant -
            (SAMPIC_MAX_NUMBER_OF_SAMPLES - digi.getCellInfo()) *
                SAMPIC_SAMPLING_PERIOD_NS;
      // End time of samples

      float firstCellTimeInstant =
          triggerCellTimeInstant -
          (SAMPIC_MAX_NUMBER_OF_SAMPLES - OFFSET_SAMPIC) *
              SAMPIC_SAMPLING_PERIOD_NS;

      std::vector<float> data;
      for (auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it)
        data.emplace_back(SAMPIC_ADC_V * (*it));

      std::vector<float> time(digi.getNumberOfSamples());
      for (unsigned int i = 0; i < time.size(); ++i)
        time.at(i) = firstCellTimeInstant + i * SAMPIC_SAMPLING_PERIOD_NS;

      auto max_it = std::max_element(data.begin(), data.end());
      auto min_it = std::min_element(data.begin(), data.end());
      // float max = std::distance(data.begin(), max_it);
      // float min = std::distance(data.begin(), min_it);
      float amplitude = *max_it - *min_it;

      float t = -1;
      t = SmartTimeOfArrival(time, data);
      if (*max_it < 0.86)
        tmp_1_ = ConstantFractionDiscriminator(time, data);

      // std::vector<float>::const_iterator max = std::max_element(data.begin(),
      // data.end());
      rec_hits.push_back(TotemTimingRecHit(
          x_pos, x_width, y_pos, y_width, z_pos, z_width, // spatial information
          t, tmp_1_, tmp_2_, amplitude));
    }
  }
}

// Double_t
// TotemTimingRecHitProducerAlgorithm::ConstantFractionDiscriminator(const
// std::vector<float>& time, const std::vector<float>& data ) {
//   int i_th=-1;
//   int i_peakStart=0;
//   float firstCrossing_tmp;
//   float firstCrossing=-1;
//   bool above=false;
//   bool lockForHysteresis=false;
//   bool peakFound=false;
//   TGraph rising;
//
//   std::vector<float>::const_iterator max = std::max_element(data.begin(),
//   data.end()); float threshold = *max * cfdFraction_;
//
//   for (unsigned int i=splinePoints_; i<data.size()-splinePoints_; ++i) {
//     // Look for first edge
//     if ( !above && !lockForHysteresis && data.at(i)>threshold ) {
//       firstCrossing_tmp = time.at(i);
//       i_peakStart = i;
//       above = true;
//       lockForHysteresis=true;
//     }
//     // Lock until above threshold+hysteresis
//     if ( above && lockForHysteresis && data.at(i)>threshold+hysteresis) {
//       lockForHysteresis=false;
//     }
//     // Look for second edge
//     if ( above && !lockForHysteresis && data.at(i)<threshold ) {
//       for (unsigned int j=0; j<splinePoints_; ++j) {
//         falling.SetPoint(j,data.at(i+j-splinePoints_/2),time.at(i+j-splinePoints_/2));
//       }
//       secondCrossing = falling.Eval(threshold,NULL,"");
//       i_th=i_peakStart;
//       peakFound=true;
//       above = false;
//     }
//
//     if (above && i>=data.size()-splinePoints_-1) {
//         if ( numberOfpeaks==0  ) secondCrossing = time.at(i);
//         peakFound=true;
//         i_th=i_peakStart;
//         above = false;
//         break;
//       }
//     if ( above && lockForHysteresis && data.at(i)<threshold &&
//     time.at(i)-firstCrossing_tmp>1e-9 ) {
//       above = false;
//       lockForHysteresis=false;
//     }
//   }
//
//   //linear interpolation
//   if ( peakFound ) {
//     for (unsigned int j=0; j<splinePoints_; ++j) {
//       rising.SetPoint(j,data.at(i_th+j-splinePoints_/2),time.at(i_th+j-splinePoints_/2));
//     }
//     firstCrossing = rising.Eval(threshold,NULL,"S");
//   }
//   return firstCrossing;
// }

TotemTimingRecHitProducerAlgorithm::RegressionResults
TotemTimingRecHitProducerAlgorithm::SimplifiedLinearRegression(
    const std::vector<float> &time, const std::vector<float> &data,
    const unsigned int start_at, const unsigned int points) const {
  RegressionResults results;
  std::cout << "start_at: " << start_at << "\tpoints" << points << std::endl;
  if (time.size() != data.size()) {
    std::cout << "Size Problem" << std::endl;
    return results;
  }
  if (start_at > data.size()) {
    std::cout << "start_at Problem" << std::endl;
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

int TotemTimingRecHitProducerAlgorithm::FastDiscriminator(
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

float TotemTimingRecHitProducerAlgorithm::SmartTimeOfArrival(
    const std::vector<float> &time, const std::vector<float> &data) {
  RegressionResults baselineRegression =
      SimplifiedLinearRegression(time, data, 0, baselinePoints_);

  //   // remove baseline
  // std::vector<float> dataCorrected( data.size() );
  // for ( unsigned int i=0; i<data.size(); ++i )
  //   dataCorrected.at(i) = data.at(i) - (baselineRegression.m * time.at(i) +
  //   baselineRegression.q);

  // float threshold = 0.4;
  float threshold =
      threholdFactor_ * baselineRegression.rms + baselineRegression.q;
  int indexOfThresholdCrossing = FastDiscriminator(data, threshold);
  std::cout << "indexOfThresholdCrossing " << indexOfThresholdCrossing
            << std::endl;
  // if ( indexOfThresholdCrossing >= 0 && indexOfThresholdCrossing < (int)
  // time.size() -1 ) return time.at(indexOfThresholdCrossing); else return -1;

  std::cout << "baselineRegression.m " << baselineRegression.m << std::endl;
  if (indexOfThresholdCrossing - risingEdgePointsBeforeTh_ < 1)
    return 0.;

  RegressionResults risingEdgeRegression = SimplifiedLinearRegression(
      time, data, indexOfThresholdCrossing - risingEdgePointsBeforeTh_,
      risingEdgePoints_);
  std::cout << "risingEdgeRegression.m " << risingEdgeRegression.m << std::endl;
  // Find intersection
  float t = (risingEdgeRegression.m * baselineRegression.q -
             baselineRegression.m * risingEdgeRegression.q) /
            (risingEdgeRegression.m - baselineRegression.m);

  tmp_2_ = risingEdgeRegression.m;

  return t;
}

float TotemTimingRecHitProducerAlgorithm::ConstantFractionDiscriminator(
    const std::vector<float> &time, const std::vector<float> &data) {
  RegressionResults baselineRegression =
      SimplifiedLinearRegression(time, data, 0, baselinePoints_);

  // remove baseline
  std::vector<float> dataCorrected(data.size());
  for (unsigned int i = 0; i < data.size(); ++i)
    dataCorrected.at(i) =
        data.at(i) - (baselineRegression.m * time.at(i) + baselineRegression.q);

  auto max_it = std::max_element(data.begin(), data.end());
  float max = *max_it;

  // float threshold = 0.4;
  float threshold = cfdFraction_ * max;
  int indexOfThresholdCrossing = FastDiscriminator(data, threshold);
  std::cout << "indexOfThresholdCrossing " << indexOfThresholdCrossing
            << std::endl;
  if (indexOfThresholdCrossing >= 0 &&
      indexOfThresholdCrossing < (int)time.size() - 1)
    return time.at(indexOfThresholdCrossing);
  else
    return -1;

  // interpolate....
}
