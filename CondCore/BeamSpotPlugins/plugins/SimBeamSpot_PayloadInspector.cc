#include "CondCore/BeamSpotPlugins/interface/BeamSpotPayloadInspectorHelper.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

namespace {

  /************************************************
    Display of Sim Beam Spot parameters
  *************************************************/

  typedef simBeamSpotPI::DisplayParameters<SimBeamSpotObjects> SimBeamSpotParameters;

  /************************************************
    Display of Sim Beam Spot parameters Differences
  *************************************************/

  typedef simBeamSpotPI::DisplayParametersDiff<SimBeamSpotObjects, cond::payloadInspector::MULTI_IOV, 1>
      SimBeamSpotParametersDiffSingleTag;
  typedef simBeamSpotPI::DisplayParametersDiff<SimBeamSpotObjects, cond::payloadInspector::SINGLE_IOV, 2>
      SimBeamSpotParametersDiffTwoTags;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SimBeamSpot) {
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotParameters);
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotParametersDiffSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotParametersDiffTwoTags);
}
