import FWCore.ParameterSet.Config as cms

process = cms.Process("TotemStandaloneRawDataTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load("EventFilter.TotemRawToDigi.totemRawToDigi_cff")

# RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

process.p = cms.Path(
    process.totemTriggerRawToDigi *
    process.totemRPRawToDigi *
    process.recoCTPPS
)

# output configuration
from RecoCTPPS.Configuration.RecoCTPPS_EventContent_cff import RecoCTPPSAOD
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:./AOD.root"),
    outputCommands = RecoCTPPSAOD.outputCommands
)

process.outpath = cms.EndPath(process.output)
