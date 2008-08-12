# import configurations
import FWCore.ParameterSet.Config as cms

# define the process
process = cms.Process("CompositeKit")

# input message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# this defines the input files
from PhysicsTools.StarterKit.RecoInput_HZZ4lRelVal_cfi import *

# input pat sequences
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")

# input composite kit sequence
process.load("PhysicsTools.StarterKit.CompositeKitDemo_cfi")

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
    fileName = cms.string('CompositeKit.root')
)

# produce Z to mu mu candidates
process.zToMuMu = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string('selectedLayer1Muons@+ selectedLayer1Muons@-'),
    cut = cms.string('0.0 < mass < 20000.0'),
    name = cms.string('zToMuMu'),
    roles = cms.vstring('muon1', 'muon2')
)

# produce Higgs to Z Z candidates
process.hToZZ = cms.EDProducer("CandViewCombiner",
    decay = cms.string('zToMuMu zToMuMu'),
    cut = cms.string('0.0 < mass < 20000.0'),
    name = cms.string('hToZZ'),
    roles = cms.vstring('Z1', 'Z2')
)

# require at least one higgs to zz candidate
process.compositeFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("hToZZ"),
    minNumber = cms.uint32(1)
)


# define event selection to be that which satisfies 'p'
process.EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

# setup event content
process.patEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
 
# define event selection to be that which satisfies 'p'
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
    fileName = cms.untracked.string('CompositeKitSkim.root')
)

# define path 'p'
process.p = cms.Path(process.patLayer0*process.patLayer1*process.zToMuMu*process.hToZZ*process.compositeFilter*process.CompositeKitDemo)
# define output path
process.outpath = cms.EndPath(process.out)
# Set the threshold for output logging to 'info'
process.MessageLogger.cerr.threshold = 'INFO'
# extend event content to include pat objects
process.patEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
# extend event content to include composite kit demo objects, and composite candidates
process.patEventContent.outputCommands.extend(['keep *_CompositeKitDemo_*_*', 'keep *_hToZZ_*_*'])

