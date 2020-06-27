import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos as _pfHiggsInteractionNetTagInfos
from RecoBTag.ONNXRuntime.pfDeepBoostedJetTags_cfi import pfDeepBoostedJetTags as _pfHiggsInteractionNetTags
from RecoBTag.ONNXRuntime.Parameters.HiggsInteractionNet.V00.pfHiggsInteractionNetPreprocessParams_cfi import pfHiggsInteractionNetPreprocessParams

# modify default parameters for tag infos
pfHiggsInteractionNetTagInfos = _pfHiggsInteractionNetTagInfos.clone(
    min_pt_for_track_properties = cms.double(0.95),
    min_puppi_wgt = cms.double(-1),
    use_puppiP4 = cms.bool(False),
    include_neutrals = cms.bool(False),
    sort_by_sip2dsig = cms.bool(True),
)

# nominal Higgs IN
pfHiggsInteractionNetTags = _pfHiggsInteractionNetTags.clone(
    src = cms.InputTag('pfHiggsInteractionNetTagInfos'),
    preprocessParams = pfHiggsInteractionNetPreprocessParams,
    model_path = cms.FileInPath('RecoBTag/Combined/data/HiggsInteractionNet/V00/IN.onnx'),
    flav_names = cms.vstring(
        'probQCD',
        'probHbb'
    ),
)

# declare all the discriminators
# nominal: probs
_pfHiggsInteractionNetTagsProbs = ['pfHiggsInteractionNetTags:' + flav_name
                       for flav_name in pfHiggsInteractionNetTags.flav_names]

