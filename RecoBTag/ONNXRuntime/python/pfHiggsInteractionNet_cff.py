import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfHiggsInteractionNetTagInfos_cfi import pfHiggsInteractionNetTagInfos
from RecoBTag.ONNXRuntime.pfHiggsInteractionNetTags_cfi import pfHiggsInteractionNetTags as _pfHiggsInteractionNetTags
from RecoBTag.ONNXRuntime.Parameters.HiggsInteractionNet.V00.pfHiggsInteractionNetPreprocessParams_cfi import pfHiggsInteractionNetPreprocessParams

# nominal Higgs IN
pfHiggsInteractionNetTags = _pfHiggsInteractionNetTags.clone(
    preprocessParams = pfHiggsInteractionNetPreprocessParams,
    model_path = 'RecoBTag/Combined/data/HiggsInteractionNet/V00/IN.onnx',
    debugMode  = False, # debug
)

# declare all the discriminators
# nominal: probs
_pfHiggsInteractionNetTagsProbs = ['pfHiggsInteractionNetTags:' + flav_name
                       for flav_name in pfHiggsInteractionNetTags.flav_names]

