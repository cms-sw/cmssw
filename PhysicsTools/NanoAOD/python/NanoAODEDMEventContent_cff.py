import FWCore.ParameterSet.Config as cms

NanoAODEDMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        "keep FlatTable_*Table_*_*",     # event data
        "keep edmTriggerResults_*_*_*",  # event data
        "keep MergableCounterTable_*Table_*_*", # accumulated per/run or per/lumi data
        "keep UniqueString_nanoMetadata_*_*",   # basic metadata
    )
)
