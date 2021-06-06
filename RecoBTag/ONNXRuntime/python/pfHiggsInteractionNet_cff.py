import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos as _pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.Parameters.HiggsInteractionNet.V00.pfHiggsInteractionNetPreprocessParams_cfi import pfHiggsInteractionNetPreprocessParams

# modify default parameters for tag infos
pfHiggsInteractionNetTagInfos = _pfDeepBoostedJetTagInfos.clone(
    min_pt_for_track_properties = 0.95,
    min_puppi_wgt = -1,
    use_puppiP4 = False,
    include_neutrals = False,
    sort_by_sip2dsig = True,
)

# nominal Higgs IN
pfHiggsInteractionNetTags = boostedJetONNXJetTagsProducer.clone(
    src = 'pfHiggsInteractionNetTagInfos',
    preprocessParams = pfHiggsInteractionNetPreprocessParams,
    model_path = 'RecoBTag/Combined/data/HiggsInteractionNet/V00/IN.onnx',
    flav_names = [ 'probQCD', 'probHbb' ]
)

# declare all the discriminators
# nominal: probs
_pfHiggsInteractionNetTagsProbs = ['pfHiggsInteractionNetTags:' + flav_name
                       for flav_name in pfHiggsInteractionNetTags.flav_names]

