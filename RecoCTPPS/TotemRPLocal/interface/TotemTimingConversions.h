/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Filip Dej
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemTimingConversions
#define RecoCTPPS_TotemRPLocal_TotemTimingConversions

#include <string>
#include <vector>
#include "TF1.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingParser.h"

class TotemTimingConversions
{
  public:
    TotemTimingConversions(bool mergeTimePeaks, const std::string& calibrationFile);

    void openCalibrationFile(const std::string& calibrationFile = "");

    float getTimeOfFirstSample(const TotemTimingDigi& digi);
    float getTriggerTime(const TotemTimingDigi& digi);
    float getTimePrecision(const TotemTimingDigi& digi);
    std::vector<float> getTimeSamples(const TotemTimingDigi& digi);
    std::vector<float> getVoltSamples(const TotemTimingDigi& digi);

  private:
    static const float SAMPIC_SAMPLING_PERIOD_NS;
    static const float SAMPIC_ADC_V;
    static const int SAMPIC_MAX_NUMBER_OF_SAMPLES;
    static const int SAMPIC_DEFAULT_OFFSET;
    static const int ACCEPTED_TIME_RADIUS;

    std::string calibrationFile_;
    bool calibrationFileOpened_;
    TotemTimingParser parsedData_;
    bool mergeTimePeaks_;
    TF1 calibrationFunction_;
};

#endif

