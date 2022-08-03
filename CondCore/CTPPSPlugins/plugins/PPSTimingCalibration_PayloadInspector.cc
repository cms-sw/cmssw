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
  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter0,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_param0;
  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter1,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_param1;
  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter2,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_param2;
  typedef ParametersPerRun<PPSTimingCalibrationPI::db0,
                           PPSTimingCalibrationPI::plane1,
                           PPSTimingCalibrationPI::channel1,
                           PPSTimingCalibrationPI::parameter3,
                           PPSTimingCalibration>
      PPSTimingCalibration_history_htdc_calibration_param3;

  /************************************************
    X-Y correlation plots
  *************************************************/

  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter0,
                         PPSTimingCalibrationPI::parameter1,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params01;
  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter0,
                         PPSTimingCalibrationPI::parameter2,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params02;
  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter0,
                         PPSTimingCalibrationPI::parameter3,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params03;

  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter1,
                         PPSTimingCalibrationPI::parameter2,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params12;
  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter1,
                         PPSTimingCalibrationPI::parameter3,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params13;

  typedef PpPCorrelation<PPSTimingCalibrationPI::db0,
                         PPSTimingCalibrationPI::plane1,
                         PPSTimingCalibrationPI::channel1,
                         PPSTimingCalibrationPI::parameter2,
                         PPSTimingCalibrationPI::parameter3,
                         PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_params23;

  /************************************************
    Image plots
  *************************************************/

  typedef ParametersPerChannel<PPSTimingCalibrationPI::db0,
                               PPSTimingCalibrationPI::plane3,
                               PPSTimingCalibrationPI::parameter3,
                               PPSTimingCalibration>
      PPSTimingCalibration_htdc_calibration_param3;
}  // namespace

PAYLOAD_INSPECTOR_MODULE(PPSTimingCalibration) {
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param0);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param1);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param2);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param3);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params01);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params02);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params03);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params12);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params13);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_params23);
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_param3);
}
