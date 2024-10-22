/*!
  \file AlignPCLThresholdsHG_PayloadInspector
  \Payload Inspector Plugin for High Granularity Alignment PCL thresholds
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/10/19 12:51:00 $
*/

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"

// for the PI Helper
#include "SiPixelAliPCLThresholdsPayloadInspectorHelper.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    Display of AlignPCLThresholds
  *************************************************/
  using AlignPCLThresholdsHG_Display =
      AlignPCLThresholdPlotHelper::AlignPCLThresholds_DisplayBase<AlignPCLThresholdsHG>;

  /************************************************
    Compare AlignPCLThresholds mapping
  *************************************************/
  using AlignPCLThresholdsHG_Compare =
      AlignPCLThresholdPlotHelper::AlignPCLThresholds_CompareBase<AlignPCLThresholdsHG, MULTI_IOV, 1>;
  using AlignPCLThresholdsHG_CompareTwoTags =
      AlignPCLThresholdPlotHelper::AlignPCLThresholds_CompareBase<AlignPCLThresholdsHG, SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(AlignPCLThresholdsHG) {
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_Display);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_Compare);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholdsHG_CompareTwoTags);
}
