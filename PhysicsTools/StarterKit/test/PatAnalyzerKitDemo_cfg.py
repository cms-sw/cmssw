import FWCore.ParameterSet.Config as cms

process = cms.Process("StarterKit")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("PhysicsTools.StarterKit.test.RecoInput_cfi")

process.load("PhysicsTools.PatAlgos.patLayer0_cff")

process.load("PhysicsTools.PatAlgos.patLayer1_cff")

process.load("PhysicsTools.StarterKit.PatAnalyzerKit_cfi")

process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(200),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('PatAnalyzerKitHistos.root')
)

process.EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)
process.patEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
process.patEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)
process.out = cms.OutputModule("PoolOutputModule",
    process.patEventSelection,
    process.patEventContent,
    verbose = cms.untracked.bool(False),
    fileName = cms.untracked.string('PatAnalyzerKitSkim.root')
)

process.p = cms.Path(process.patLayer0*process.patLayer1*process.patAnalyzerKit)
process.outpath = cms.EndPath(process.out)
process.MessageLogger.cerr.threshold = 'INFO'
process.patEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
process.patEventContent.outputCommands.extend(['keep *_patAnalyzerKit_*_*'])

