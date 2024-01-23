
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

  using PPSTimingCalibration_history_htdc_calibration_param0 =
      ParametersPerRun<PPSTimingCalibrationPI::parameter0, PPSTimingCalibration>;
  using PPSTimingCalibration_history_htdc_calibration_param1 =
      ParametersPerRun<PPSTimingCalibrationPI::parameter1, PPSTimingCalibration>;
  using PPSTimingCalibration_history_htdc_calibration_param2 =
      ParametersPerRun<PPSTimingCalibrationPI::parameter2, PPSTimingCalibration>;
  using PPSTimingCalibration_history_htdc_calibration_param3 =
      ParametersPerRun<PPSTimingCalibrationPI::parameter3, PPSTimingCalibration>;

  /************************************************
    Image plots
  *************************************************/

  using PPSTimingCalibration_htdc_calibration_param0_per_channels =
      ParametersPerChannel<PPSTimingCalibrationPI::parameter0, PPSTimingCalibration>;
  using PPSTimingCalibration_htdc_calibration_param1_per_channels =
      ParametersPerChannel<PPSTimingCalibrationPI::parameter1, PPSTimingCalibration>;
  using PPSTimingCalibration_htdc_calibration_param2_per_channels =
      ParametersPerChannel<PPSTimingCalibrationPI::parameter2, PPSTimingCalibration>;
  using PPSTimingCalibration_htdc_calibration_param3_per_channels =
      ParametersPerChannel<PPSTimingCalibrationPI::parameter3, PPSTimingCalibration>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(PPSTimingCalibration) {
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param0)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param1)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param2)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_history_htdc_calibration_param3)

  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_param0_per_channels)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_param1_per_channels)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_param2_per_channels)
  PAYLOAD_INSPECTOR_CLASS(PPSTimingCalibration_htdc_calibration_param3_per_channels)
}