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
#include <boost/exception/exception.hpp>

//----------------------------------------------------------------------------------------------------

const float TotemTimingConversions::SAMPIC_SAMPLING_PERIOD_NS = 1. / 7.695;
const float TotemTimingConversions::SAMPIC_ADC_V = 1. / 256;
const int TotemTimingConversions::SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
const int TotemTimingConversions::SAMPIC_DEFAULT_OFFSET = 30;
const int TotemTimingConversions::ACCEPTED_TIME_RADIUS = 4;

//----------------------------------------------------------------------------------------------------

TotemTimingConversions::TotemTimingConversions(bool mergeTimePeaks, const std::string& calibrationFile) :
  calibrationFile_(calibrationFile),
  calibrationFileOpened_(false),
  mergeTimePeaks_(mergeTimePeaks)
{}

//----------------------------------------------------------------------------------------------------

void
TotemTimingConversions::openCalibrationFile(const std::string& calibrationFile)
{
  if (!calibrationFile.empty())
    calibrationFile_ = calibrationFile;
  if (calibrationFile_.empty())
    return;

  try {
    parsedData_.parseFile(calibrationFile_);
  } catch (const boost::exception& e) { return; }
  calibrationFunction_ = TF1("calibrationFunction_", parsedData_.getFormula().c_str());
  calibrationFileOpened_ = true;
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
    ((digi.getFPGATimestamp()-timestamp) & 0xFFFFFFF000)
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
  if (!calibrationFileOpened_)
    openCalibrationFile();
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
  if (!calibrationFileOpened_)
    openCalibrationFile();

  unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
  if (offsetOfSamples == 0)
    offsetOfSamples = 30; // FW 0 is not sending this, FW > 0 yes

  return getTimeOfFirstSample(digi) + (SAMPIC_MAX_NUMBER_OF_SAMPLES - offsetOfSamples) * SAMPIC_SAMPLING_PERIOD_NS;
}

//----------------------------------------------------------------------------------------------------

float
TotemTimingConversions::getTimePrecision(const TotemTimingDigi& digi)
{
  if (!calibrationFileOpened_)
    openCalibrationFile();
  int db = digi.getHardwareBoardId();
  int sampic = digi.getHardwareSampicId();
  int channel = digi.getHardwareChannelId();
  return parsedData_.getTimePrecision(db, sampic, channel);
}

//----------------------------------------------------------------------------------------------------

std::vector<float>
TotemTimingConversions::getTimeSamples(const TotemTimingDigi& digi)
{
  if (!calibrationFileOpened_)
    openCalibrationFile();

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
  if (!calibrationFileOpened_)
    openCalibrationFile();

  std::vector<float> data;
  if (calibrationFile_.empty())
    for (auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it)
      data.emplace_back(SAMPIC_ADC_V * (*it));
  else {
    unsigned int db = digi.getHardwareBoardId();
    unsigned int sampic = digi.getHardwareSampicId();
    unsigned int channel = digi.getHardwareChannelId();
    unsigned int cell = digi.getCellInfo();
    for (auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it, ++cell){
      auto parameters = parsedData_.getParameters(db, sampic, channel, cell);
      for (unsigned int i=0; i<parameters.size(); ++i)
        calibrationFunction_.SetParameter(i, parameters.at(i));
      data.emplace_back(calibrationFunction_.Eval(*it));
    }
  }
  return data;
}

