import FWCore.ParameterSet.Config as cms

NanoAODEDMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        "keep *_genWeightsTable_*_*",     # event data
        "keep nanoaodFlatTable_*Table_*_*",     # event data
        "keep nanoaodFlatTables_*Table_*_*",     # event data
        "keep edmTriggerResults_*_*_*",  # event data
        "keep String_*_genModel_*",  # generator model data
        "keep nanoaodMergeableCounterTable_*Table_*_*", # accumulated per/run or per/lumi data
        "keep nanoaodUniqueString_nanoMetadata_*_*",   # basic metadata
    )
)

NANOAODEventContent = NanoAODEDMEventContent.clone(
    compressionLevel = cms.untracked.int32(9),
    compressionAlgorithm = cms.untracked.string("LZMA"),
)
NANOAODSIMEventContent = NanoAODEDMEventContent.clone(
    compressionLevel = cms.untracked.int32(9),
    compressionAlgorithm = cms.untracked.string("LZMA"),
)
NANOAODGENEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        "keep *_genWeightsTable_*_*",     # event data
        "keep nanoaodFlatTable_*Table_*_*",     # event data
        "keep String_*_genModel_*",  # generator model data
        "keep nanoaodMergeableCounterTable_*Table_*_*", # accumulated per/run or per/lumi data
        "keep nanoaodUniqueString_nanoMetadata_*_*",   # basic metadata
    ),
    compressionLevel = cms.untracked.int32(9),
    compressionAlgorithm = cms.untracked.string("LZMA"),
)
