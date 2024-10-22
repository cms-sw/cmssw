/****************************************************************************
 *
 * This is a part of PPS PI software.
 *
 ****************************************************************************/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondCore/CTPPSPlugins/interface/DAQMappingPayloadInspectorHelper.h"

namespace {
  typedef DAQMappingPayloadInfo<TotemDAQMapping> DAQMappingPayloadInfo_Text;
}

PAYLOAD_INSPECTOR_MODULE(TotemDAQMapping) { PAYLOAD_INSPECTOR_CLASS(DAQMappingPayloadInfo_Text); }
