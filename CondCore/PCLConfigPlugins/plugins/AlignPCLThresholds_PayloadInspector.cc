/*!
  \file AlignPCLThresholds_PayloadInspector
  \Payload Inspector Plugin for Alignment PCL thresholds
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/10/19 12:51:00 $
*/

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"

// for the PI Helper
#include "SiPixelAliPCLThresholdsPayloadInspectorHelper.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    Display of AlignPCLThresholds
  *************************************************/
  using AlignPCLThresholds_Display = AlignPCLThresholdPlotHelper::AlignPCLThresholds_DisplayBase<AlignPCLThresholds>;

  /************************************************
    Compare AlignPCLThresholds mapping
  *************************************************/
  using AlignPCLThresholds_Compare =
      AlignPCLThresholdPlotHelper::AlignPCLThresholds_CompareBase<AlignPCLThresholds, MULTI_IOV, 1>;
  using AlignPCLThresholds_CompareTwoTags =
      AlignPCLThresholdPlotHelper::AlignPCLThresholds_CompareBase<AlignPCLThresholds, SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(AlignPCLThresholds) {
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholds_Display);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholds_Compare);
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholds_CompareTwoTags);
}
