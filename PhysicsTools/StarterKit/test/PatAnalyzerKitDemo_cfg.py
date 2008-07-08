# Import configurations
import FWCore.ParameterSet.Config as cms

# set up process
process = cms.Process("StarterKit")

# this defines the input files
from PhysicsTools.StarterKit.RecoInput_cfi import *

# input message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# input pat sequences
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")

# input pat analyzer sequence
process.load("PhysicsTools.StarterKit.PatAnalyzerKit_cfi")

# load the pat layer 1 event content
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")

# request a summary at the end of the file
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# define the source, from reco input
process.source = RecoInput()

# set the number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# talk to TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('PatAnalyzerKitHistos.root')
)

# define event selection to be that which satisfies 'p'
process.patEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)


# define path 'p'
process.p = cms.Path(process.patLayer0*process.patLayer1*process.patAnalyzerKit)
# Set the threshold for output logging to 'info'
process.MessageLogger.cerr.threshold = 'INFO'
# extend event content to include pat analyzer kit objects
process.patLayer1EventContent.outputCommands.extend(['keep *_patAnalyzerKit_*_*'])


# talk to output module
process.out = cms.OutputModule("PoolOutputModule",
    process.patEventSelection,
    process.patLayer1EventContent,
    verbose = cms.untracked.bool(False),
    fileName = cms.untracked.string('PatAnalyzerKitSkim.root')
)


# define output path
process.outpath = cms.EndPath(process.out)
