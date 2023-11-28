import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos as _pfDeepBoostedJetTagInfos
from RecoBTag.ONNXRuntime.boostedJetONNXJetTagsProducer_cfi import boostedJetONNXJetTagsProducer
from RecoBTag.ONNXRuntime.Parameters.HiggsInteractionNet.V00.pfHiggsInteractionNetPreprocessParams_cfi import pfHiggsInteractionNetPreprocessParams
from RecoBTag.ONNXRuntime.particleNetSonicJetTagsProducer_cfi import particleNetSonicJetTagsProducer as _particleNetSonicJetTagsProducer
from Configuration.ProcessModifiers.particleNetSonicTriton_cff import particleNetSonicTriton

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
    model_path = 'RecoBTag/Combined/data/HiggsInteractionNet/V00/modelfile/model.onnx',
    flav_names = [ 'probQCD', 'probHbb' ]
)

particleNetSonicTriton.toReplaceWith(pfHiggsInteractionNetTags, _particleNetSonicJetTagsProducer.clone(
    src = 'pfHiggsInteractionNetTagInfos',
    preprocess_json = 'RecoBTag/Combined/data/models/higgsInteractionNet/preprocess.json',
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("higgsInteractionNet"),
        modelConfigPath = cms.FileInPath("RecoBTag/Combined/data/models/higgsInteractionNet/config.pbtxt"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    flav_names = pfHiggsInteractionNetTags.flav_names,
))

# declare all the discriminators
# nominal: probs
_pfHiggsInteractionNetTagsProbs = ['pfHiggsInteractionNetTags:' + flav_name
                       for flav_name in pfHiggsInteractionNetTags.flav_names]

