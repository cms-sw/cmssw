#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondCore/CTPPSPlugins/interface/PPSTimingCalibrationPayloadInspectorHelper.h"
#include <memory>
#include <sstream>

#include "TH2D.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"

namespace {

  /************************************************
    History plots
  *************************************************/

  //db=0, plane=0, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param3;

  //db=0, plane=0, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param3;

  //db=0, plane=0, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param3;

  //db=0, plane=0, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param3;

  //db=0, plane=0, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param3;

  //db=0, plane=0, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param3;

  //db=0, plane=0, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param3;

  //db=0, plane=0, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param3;

  //db=0, plane=0, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param3;

  //db=0, plane=0, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param3;

  //db=0, plane=0, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param3;

  //db=0, plane=0, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param3;

  //db=0, plane=1, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param3;

  //db=0, plane=1, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param3;

  //db=0, plane=1, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param3;

  //db=0, plane=1, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param3;

  //db=0, plane=1, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param3;

  //db=0, plane=1, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param3;

  //db=0, plane=1, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param3;

  //db=0, plane=1, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param3;

  //db=0, plane=1, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param3;

  //db=0, plane=1, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param3;

  //db=0, plane=1, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param3;

  //db=0, plane=1, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param3;

  //db=0, plane=2, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param3;

  //db=0, plane=2, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param3;

  //db=0, plane=2, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param3;

  //db=0, plane=2, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param3;

  //db=0, plane=2, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param3;

  //db=0, plane=2, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param3;

  //db=0, plane=2, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param3;

  //db=0, plane=2, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param3;

  //db=0, plane=2, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param3;

  //db=0, plane=2, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param3;

  //db=0, plane=2, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param3;

  //db=0, plane=2, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param3;

  //db=0, plane=3, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param3;

  //db=0, plane=3, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param3;

  //db=0, plane=3, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param3;

  //db=0, plane=3, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param3;

  //db=0, plane=3, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param3;

  //db=0, plane=3, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param3;

  //db=0, plane=3, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param3;

  //db=0, plane=3, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param3;

  //db=0, plane=3, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param3;

  //db=0, plane=3, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param3;

  //db=0, plane=3, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param3;

  //db=0, plane=3, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param3;

  //db=1, plane=0, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param3;

  //db=1, plane=0, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param3;

  //db=1, plane=0, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param3;

  //db=1, plane=0, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param3;

  //db=1, plane=0, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param3;

  //db=1, plane=0, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param3;

  //db=1, plane=0, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param3;

  //db=1, plane=0, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param3;

  //db=1, plane=0, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param3;

  //db=1, plane=0, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param3;

  //db=1, plane=0, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param3;

  //db=1, plane=0, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane0,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param3;

  //db=1, plane=1, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param3;

  //db=1, plane=1, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param3;

  //db=1, plane=1, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param3;

  //db=1, plane=1, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param3;

  //db=1, plane=1, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param3;

  //db=1, plane=1, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param3;

  //db=1, plane=1, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param3;

  //db=1, plane=1, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param3;

  //db=1, plane=1, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param3;

  //db=1, plane=1, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param3;

  //db=1, plane=1, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param3;

  //db=1, plane=1, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param3;

  //db=1, plane=2, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param3;

  //db=1, plane=2, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param3;

  //db=1, plane=2, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param3;

  //db=1, plane=2, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param3;

  //db=1, plane=2, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param3;

  //db=1, plane=2, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param3;

  //db=1, plane=2, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param3;

  //db=1, plane=2, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param3;

  //db=1, plane=2, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param3;

  //db=1, plane=2, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param3;

  //db=1, plane=2, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param3;

  //db=1, plane=2, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane2,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param3;

  //db=1, plane=3, channel=0

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel0,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param3;

  //db=1, plane=3, channel=1

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param3;

  //db=1, plane=3, channel=2

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel2,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param3;

  //db=1, plane=3, channel=3

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel3,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param3;

  //db=1, plane=3, channel=4

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel4,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param3;

  //db=1, plane=3, channel=5

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel5,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param3;

  //db=1, plane=3, channel=6

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel6,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param3;

  //db=1, plane=3, channel=7

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel7,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param3;

  //db=1, plane=3, channel=8

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel8,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param3;

  //db=1, plane=3, channel=9

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel9,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param3;

  //db=1, plane=3, channel=10

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel10,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param3;

  //db=1, plane=3, channel=11

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param0;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param1;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param2;

  typedef ParametersPerRun<PPSTimingCalibrationPI::db1,
                           PPSTimingCalibrationPI::plane3,
                           PPSTimingCalibrationPI::channel11,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param3;

  /************************************************
    X-Y correlation plots
  *************************************************/

  /************************************************
    Image plots
  *************************************************/

  //db=0, plane=0

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl0param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl0param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl0param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl0param3_per_channels;

  //db=0, plane=1

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl1param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl1param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl1param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl1param3_per_channels;

  //db=0, plane=2

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl2param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl2param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl2param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl2param3_per_channels;

  //db=0, plane=3

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl3param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl3param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl3param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db0pl3param3_per_channels;

  //db=1, plane=0

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl0param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl0param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl0param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane0,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl0param3_per_channels;

  //db=1, plane=1

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl1param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl1param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl1param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane1,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl1param3_per_channels;

  //db=1, plane=2

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl2param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl2param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl2param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane2,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl2param3_per_channels;

  //db=1, plane=3

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter0,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl3param0_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter1,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl3param1_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter2,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl3param2_per_channels;

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db1,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_db1pl3param3_per_channels;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(PPSTimingCalibration) {
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl0ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl1ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl2ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db0pl3ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl0ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl1ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl2ch11_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch0_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch1_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch2_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch3_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch4_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch5_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch6_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch7_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch8_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch9_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch10_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_db1pl3ch11_param3);

  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl0param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl0param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl0param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl0param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl1param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl1param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl1param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl1param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl2param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl2param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl2param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl2param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl3param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl3param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl3param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db0pl3param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl0param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl0param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl0param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl0param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl1param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl1param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl1param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl1param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl2param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl2param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl2param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl2param3_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl3param0_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl3param1_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl3param2_per_channels);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_db1pl3param3_per_channels);
}
