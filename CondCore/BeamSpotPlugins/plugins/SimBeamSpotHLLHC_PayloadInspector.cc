#include "CondCore/BeamSpotPlugins/interface/BeamSpotPayloadInspectorHelper.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"

namespace {

  /************************************************
    Display of Sim Beam Spot HL-LHC parameters
  *************************************************/

  typedef simBeamSpotHLLHCPI::DisplayParameters<SimBeamSpotHLLHCObjects> SimBeamSpotHLLHCParameters;

  /*********************************************************
    Display of Sim Beam Spot HL-LHC parameters Differences
  **********************************************************/

  typedef simBeamSpotHLLHCPI::DisplayParametersDiff<SimBeamSpotHLLHCObjects, cond::payloadInspector::MULTI_IOV, 1>
      SimBeamSpotHLLHCParametersDiffSingleTag;
  typedef simBeamSpotHLLHCPI::DisplayParametersDiff<SimBeamSpotHLLHCObjects, cond::payloadInspector::SINGLE_IOV, 2>
      SimBeamSpotHLLHCParametersDiffTwoTags;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SimBeamSpotHLLHC) {
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotHLLHCParameters);
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotHLLHCParametersDiffSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SimBeamSpotHLLHCParametersDiffTwoTags);
}
