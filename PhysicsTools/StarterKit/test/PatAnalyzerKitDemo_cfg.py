# Import configurations
import FWCore.ParameterSet.Config as cms

# set up process
process = cms.Process("StarterKit")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('PATLayer0Summary')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    PATLayer0Summary = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Load geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# this defines the input files
from PhysicsTools.StarterKit.RecoInput_cfi import *

# this inputs the input files from the previous function
process.source = RecoInput()

# set the number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)


# input pat sequences
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")

# input pat analyzer sequence
process.load("PhysicsTools.StarterKit.PatAnalyzerKit_cfi")

# talk to TFileService for output histograms
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('PatAnalyzerKitHistos.root')
)

# define path 'p': PAT Layer 0, PAT Layer 1, and the analyzer
process.p = cms.Path(process.patLayer0*
                     process.patLayer1*
                     process.patAnalyzerKit)


# load the pat layer 1 event content
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")

# setup event content: drop everything before PAT
process.patEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)

# extend event content to include PAT objects
process.patEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)

# extend event content to include pat analyzer kit objects from EDNtuple
process.patLayer1EventContent.outputCommands.extend(['keep *_patAnalyzerKit_*_*'])

# define output event selection to be that which satisfies 'p'
process.patEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

# talk to output module
process.out = cms.OutputModule("PoolOutputModule",
    process.patEventSelection,
    process.patEventContent,
    verbose = cms.untracked.bool(False),
    fileName = cms.untracked.string('PatAnalyzerKitSkim.root')
)

# define output path
process.outpath = cms.EndPath(process.out)

