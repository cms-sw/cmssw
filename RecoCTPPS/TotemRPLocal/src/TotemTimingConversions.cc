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
  unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = SAMPIC_DEFAULT_OFFSET;  // FW 0 is not sending this, FW > 0 yes

  unsigned int timestamp =
      (digi.getCellInfo() <= SAMPIC_MAX_NUMBER_OF_SAMPLES / 2) ? digi.getTimestampA() : digi.getTimestampB();

  int cell0TimeClock = timestamp + ((digi.getFPGATimestamp() - timestamp) & CELL0_MASK) -
                       digi.getEventInfo().getL1ATimestamp() + digi.getEventInfo().getL1ALatency();

  // time of first cell
  float cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

  // time of triggered cell
  float firstCellTimeInstant =
      (digi.getCellInfo() < offsetOfSamples)
          ? cell0TimeInstant + digi.getCellInfo() * SAMPIC_SAMPLING_PERIOD_NS
          : cell0TimeInstant - (SAMPIC_MAX_NUMBER_OF_SAMPLES - digi.getCellInfo()) * SAMPIC_SAMPLING_PERIOD_NS;

  int db = digi.getHardwareBoardId();
  int sampic = digi.getHardwareSampicId();
  int channel = digi.getHardwareChannelId();
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
  unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = 30;  // FW 0 is not sending this, FW > 0 yes

  return timeOfFirstSample(digi) + (SAMPIC_MAX_NUMBER_OF_SAMPLES - offsetOfSamples) * SAMPIC_SAMPLING_PERIOD_NS;
}

//----------------------------------------------------------------------------------------------------

float TotemTimingConversions::timePrecision(const TotemTimingDigi& digi) const {
  int db = digi.getHardwareBoardId();
  int sampic = digi.getHardwareSampicId();
  int channel = digi.getHardwareChannelId();
  return calibration_.timePrecision(db, sampic, channel);
}

//----------------------------------------------------------------------------------------------------

std::vector<float> TotemTimingConversions::timeSamples(const TotemTimingDigi& digi) const {
  std::vector<float> time(digi.getNumberOfSamples());
  for (unsigned int i = 0; i < time.size(); ++i)
    time.at(i) = timeOfFirstSample(digi) + i * SAMPIC_SAMPLING_PERIOD_NS;
  return time;
}

//----------------------------------------------------------------------------------------------------
// NOTE: If no proper file is specified, calibration is not applied

std::vector<float> TotemTimingConversions::voltSamples(const TotemTimingDigi& digi) const {
  std::vector<float> data;
  if (calibrationFunction_.numberOfVariables() != 1)
    for (const auto& sample : digi.getSamples())
      data.emplace_back(SAMPIC_ADC_V * sample);
  else {
    unsigned int db = digi.getHardwareBoardId();
    unsigned int sampic = digi.getHardwareSampicId();
    unsigned int channel = digi.getHardwareChannelId();
    unsigned int cell = digi.getCellInfo();
    for (const auto& sample : digi.getSamples()) {
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
