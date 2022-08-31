/****************************************************************************
 *
 * This is a part of PPS PI software.
 *
 ****************************************************************************/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondCore/CTPPSPlugins/interface/PPSAlignmentConfigurationHelper.h"

namespace {
  typedef AlignmentPayloadInfo<PPSAlignmentConfiguration> PPSAlignmentConfig_Payload_TextInfo;
}

PAYLOAD_INSPECTOR_MODULE(PPSAlignmentConfiguration) { PAYLOAD_INSPECTOR_CLASS(PPSAlignmentConfig_Payload_TextInfo); }