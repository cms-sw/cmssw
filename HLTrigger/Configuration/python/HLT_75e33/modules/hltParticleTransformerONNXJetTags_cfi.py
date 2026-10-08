import FWCore.ParameterSet.Config as cms

hltParticleTransformerONNXJetTags = cms.EDProducer( "HLTParticleTransformerAK4ONNXJetTagsProducer",
    src = cms.InputTag( "hltParticleTransformerAK4TagInfos" ),
    model_path = cms.FileInPath( "RecoBTag/Combined/data/HLT/hltParticleTransformerAK4/hltParTAK4_CMSSW15_082026.onnx" ),
    flav_names = cms.vstring(
        "probb",
        "probc",
        "probuds",
        "probg",
        "probtaup",
        "probtaum",
    ),
    input_names = cms.vstring(
        "global",
        "cpf",
        "vtx",
    ),
    output_names = cms.vstring("output"),
    mightGet = cms.optional.untracked.vstring,
)
