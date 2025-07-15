import FWCore.ParameterSet.Config as cms

hltPfDeepFlavourJetTags = cms.EDProducer("DeepFlavourONNXJetTagsProducer",
    flav_names = cms.vstring(
        'probb',
        'probbb',
        'problepb',
        'probc',
        'probuds',
        'probg'
    ),
    input_names = cms.vstring(
        'input_1',
        'input_2',
        'input_3',
        'input_4',
        'input_5'
    ),
    mightGet = cms.optional.untracked.vstring,
    model_path = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourV03_PhaseII/Deepjet_FixedSoftmaxLayer.onnx'),
    output_names = cms.vstring(),
    src = cms.InputTag("hltPfDeepFlavourTagInfos"),
    produceValueMap = cms.untracked.bool(True),
    jets = cms.InputTag("hltAK4PFPuppiJets")
)
