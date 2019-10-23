/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Filip Dej
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingConversions.h"
#include "FWCore/Utilities/interface/Exception.h"

//----------------------------------------------------------------------------------------------------

TotemTimingConversions::TotemTimingConversions(bool mergeTimePeaks, const PPSTimingCalibration& calibration)
    : calibration_(calibration), mergeTimePeaks_(mergeTimePeaks), calibrationFunction_(calibration_.formula()) {}

//----------------------------------------------------------------------------------------------------

float TotemTimingConversions::timeOfFirstSample(const TotemTimingDigi& digi) const {
  unsigned int offsetOfSamples = digi.eventInfo().offsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = SAMPIC_DEFAULT_OFFSET;  // FW 0 is not sending this, FW > 0 yes

  unsigned int timestamp =
      (digi.cellInfo() <= SAMPIC_MAX_NUMBER_OF_SAMPLES / 2) ? digi.timestampA() : digi.timestampB();

  int cell0TimeClock = timestamp + ((digi.fpgaTimestamp() - timestamp) & CELL0_MASK) - digi.eventInfo().l1ATimestamp() +
                       digi.eventInfo().l1ALatency();

  // time of first cell
  float cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

  // time of triggered cell
  float firstCellTimeInstant =
      (digi.cellInfo() < offsetOfSamples)
          ? cell0TimeInstant + digi.cellInfo() * SAMPIC_SAMPLING_PERIOD_NS
          : cell0TimeInstant - (SAMPIC_MAX_NUMBER_OF_SAMPLES - digi.cellInfo()) * SAMPIC_SAMPLING_PERIOD_NS;

  int db = digi.hardwareBoardId();
  int sampic = digi.hardwareSampicId();
  int channel = digi.hardwareChannelId();
  float t = firstCellTimeInstant + calibration_.timeOffset(db, sampic, channel);
  //NOTE: If no time offset is set, timeOffset returns 0

  if (mergeTimePeaks_) {
    if (t < -ACCEPTED_TIME_RADIUS)
      t += SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS;
    if (t > ACCEPTED_TIME_RADIUS)
      t -= SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS;
  }
  return t;
}

//----------------------------------------------------------------------------------------------------

float TotemTimingConversions::triggerTime(const TotemTimingDigi& digi) const {
  unsigned int offsetOfSamples = digi.eventInfo().offsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = 30;  // FW 0 is not sending this, FW > 0 yes

  return timeOfFirstSample(digi) + (SAMPIC_MAX_NUMBER_OF_SAMPLES - offsetOfSamples) * SAMPIC_SAMPLING_PERIOD_NS;
}

//----------------------------------------------------------------------------------------------------

float TotemTimingConversions::timePrecision(const TotemTimingDigi& digi) const {
  int db = digi.hardwareBoardId();
  int sampic = digi.hardwareSampicId();
  int channel = digi.hardwareChannelId();
  return calibration_.timePrecision(db, sampic, channel);
}

//----------------------------------------------------------------------------------------------------

std::vector<float> TotemTimingConversions::timeSamples(const TotemTimingDigi& digi) const {
  std::vector<float> time(digi.numberOfSamples());
  for (unsigned int i = 0; i < time.size(); ++i)
    time.at(i) = timeOfFirstSample(digi) + i * SAMPIC_SAMPLING_PERIOD_NS;
  return time;
}

//----------------------------------------------------------------------------------------------------
// NOTE: If no proper file is specified, calibration is not applied

std::vector<float> TotemTimingConversions::voltSamples(const TotemTimingDigi& digi) const {
  std::vector<float> data;
  if (calibrationFunction_.numberOfVariables() != 1)
    for (const auto& sample : digi.samples())
      data.emplace_back(SAMPIC_ADC_V * sample);
  else {
    unsigned int db = digi.hardwareBoardId();
    unsigned int sampic = digi.hardwareSampicId();
    unsigned int channel = digi.hardwareChannelId();
    unsigned int cell = digi.cellInfo();
    for (const auto& sample : digi.samples()) {
      // ring buffer on Sampic, so accounting for samples register boundary
      const unsigned short sample_cell = (cell++) % SAMPIC_MAX_NUMBER_OF_SAMPLES;
      auto parameters = calibration_.parameters(db, sampic, channel, sample_cell);
      if (parameters.empty() || parameters.size() != calibrationFunction_.numberOfParameters())
        throw cms::Exception("TotemTimingConversions:voltSamples")
            << "Invalid calibrations retrieved for Sampic digi"
            << " (" << db << ", " << sampic << ", " << channel << ", " << sample_cell << ")!";
      data.emplace_back(calibrationFunction_.evaluate(std::vector<double>{(double)sample}, parameters));
    }
  }
  return data;
}
