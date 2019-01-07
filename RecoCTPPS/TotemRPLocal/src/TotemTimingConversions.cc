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

#include <boost/exception/exception.hpp>

//----------------------------------------------------------------------------------------------------

const float TotemTimingConversions::SAMPIC_SAMPLING_PERIOD_NS = 1. / 7.695;
const float TotemTimingConversions::SAMPIC_ADC_V = 1. / 256;
const int TotemTimingConversions::SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
const int TotemTimingConversions::SAMPIC_DEFAULT_OFFSET = 30;
const int TotemTimingConversions::ACCEPTED_TIME_RADIUS = 4;
const unsigned long TotemTimingConversions::CELL0_MASK = 0xfffffff000;

//----------------------------------------------------------------------------------------------------

TotemTimingConversions::TotemTimingConversions(bool mergeTimePeaks, const std::string& calibrationFile) :
  calibrationFileOpened_(false),
  mergeTimePeaks_(mergeTimePeaks)
{
  if (!calibrationFile.empty())
    try {
      parsedData_.parseFile(calibrationFile);
      calibrationFunction_ = TF1("calibrationFunction_", parsedData_.getFormula().c_str());
      calibrationFileOpened_ = true;
    } catch (const boost::exception& e) {
      throw cms::Exception("TotemTimingConversions")
        << "Failed to open calibration file \"" << calibrationFile << "\" for parsing!";
    }
}

//----------------------------------------------------------------------------------------------------

float
TotemTimingConversions::getTimeOfFirstSample(const TotemTimingDigi& digi)
{
  unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = SAMPIC_DEFAULT_OFFSET; // FW 0 is not sending this, FW > 0 yes

  unsigned int timestamp = (digi.getCellInfo() <= SAMPIC_MAX_NUMBER_OF_SAMPLES/2)
    ? digi.getTimestampA()
    : digi.getTimestampB();

  int cell0TimeClock = timestamp +
    ((digi.getFPGATimestamp()-timestamp) & CELL0_MASK)
    - digi.getEventInfo().getL1ATimestamp()
    + digi.getEventInfo().getL1ALatency();

  // time of first cell
  float cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

  // time of triggered cell
  float firstCellTimeInstant = (digi.getCellInfo() < offsetOfSamples)
    ? cell0TimeInstant + digi.getCellInfo() * SAMPIC_SAMPLING_PERIOD_NS
    : cell0TimeInstant - (SAMPIC_MAX_NUMBER_OF_SAMPLES - digi.getCellInfo())*SAMPIC_SAMPLING_PERIOD_NS;

  int db = digi.getHardwareBoardId();
  int sampic = digi.getHardwareSampicId();
  int channel = digi.getHardwareChannelId();
  float t = firstCellTimeInstant + parsedData_.getTimeOffset(db, sampic, channel);
  //NOTE: If no time offset is set, getTimeOffset returns 0

  if (mergeTimePeaks_) {
    if (t < -ACCEPTED_TIME_RADIUS)
      t += SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS;
    if (t > ACCEPTED_TIME_RADIUS)
      t -= SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS;
  }
  return t;
}

//----------------------------------------------------------------------------------------------------

float
TotemTimingConversions::getTriggerTime(const TotemTimingDigi& digi)
{
  unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = 30; // FW 0 is not sending this, FW > 0 yes

  return getTimeOfFirstSample(digi) + (SAMPIC_MAX_NUMBER_OF_SAMPLES - offsetOfSamples) * SAMPIC_SAMPLING_PERIOD_NS;
}

//----------------------------------------------------------------------------------------------------

float
TotemTimingConversions::getTimePrecision(const TotemTimingDigi& digi)
{
  int db = digi.getHardwareBoardId();
  int sampic = digi.getHardwareSampicId();
  int channel = digi.getHardwareChannelId();
  return parsedData_.getTimePrecision(db, sampic, channel);
}

//----------------------------------------------------------------------------------------------------

std::vector<float>
TotemTimingConversions::getTimeSamples(const TotemTimingDigi& digi)
{
  std::vector<float> time(digi.getNumberOfSamples());
  for (unsigned int i = 0; i < time.size(); ++i)
    time.at(i) = getTimeOfFirstSample(digi) + i * SAMPIC_SAMPLING_PERIOD_NS;
  return time;
}

//----------------------------------------------------------------------------------------------------
// NOTE: If no proper file is specified, calibration is not applied

std::vector<float>
TotemTimingConversions::getVoltSamples(const TotemTimingDigi& digi)
{
  std::vector<float> data;
  if (!calibrationFileOpened_)
    for (const auto& sample : digi.getSamples())
      data.emplace_back(SAMPIC_ADC_V * sample);
  else {
    unsigned int db = digi.getHardwareBoardId();
    unsigned int sampic = digi.getHardwareSampicId();
    unsigned int channel = digi.getHardwareChannelId();
    unsigned int cell = digi.getCellInfo();
    for (const auto& sample : digi.getSamples()) {
      auto parameters = parsedData_.getParameters(db, sampic, channel, ++cell);
      if (parameters.empty())
        throw cms::Exception("TotemTimingConversions:getVoltSamples")
          << "Invalid calibrations retrieved for Sampic digi"
          << " (" << db << ", " << sampic << ", " << channel << ", " << cell << ")!";
      double x = (double)sample;
      data.emplace_back(calibrationFunction_.EvalPar(&x, parameters.data()));
    }
  }
  return data;
}

