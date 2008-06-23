import FWCore.ParameterSet.Config as cms

process = cms.Process("CompositeKit")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("PhysicsTools.StarterKit.test.RecoInput_HZZ4lRelVal_cfi")

process.load("PhysicsTools.PatAlgos.patLayer0_cff")

process.load("PhysicsTools.PatAlgos.patLayer1_cff")

process.load("PhysicsTools.StarterKit.CompositeKitDemo_cfi")

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
    fileName = cms.string('CompositeKit.root')
)

process.zToMuMu = cms.EDFilter("NamedCandViewShallowCloneCombiner",
    cut = cms.string('0.0 < mass < 20000.0'),
    name = cms.string('zToMuMu'),
    roles = cms.vstring('muon1', 
        'muon2'),
    decay = cms.string('selectedLayer1Muons@+ selectedLayer1Muons@-')
)

process.hToZZ = cms.EDFilter("NamedCandViewCombiner",
    cut = cms.string('0.0 < mass < 20000.0'),
    name = cms.string('hToZZ'),
    roles = cms.vstring('Z1', 
        'Z2'),
    decay = cms.string('zToMuMu zToMuMu')
)

process.compositeFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("hToZZ"),
    minNumber = cms.uint32(1)
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
    fileName = cms.untracked.string('CompositeKitSkim.root')
)

process.p = cms.Path(process.patLayer0*process.patLayer1*process.zToMuMu*process.hToZZ*process.compositeFilter*process.CompositeKitDemo)
process.outpath = cms.EndPath(process.out)
process.MessageLogger.cerr.threshold = 'INFO'
process.patEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
process.patEventContent.outputCommands.extend(['keep *_CompositeKitDemo_*_*', 'keep *_hToZZ_*_*'])

