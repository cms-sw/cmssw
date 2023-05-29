import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    PATSummaryTables = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/1E2DDC37-EC1A-DE11-BB33-0030487F933D.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/32BEC84F-DE1A-DE11-9602-0030487EB003.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/32EB7039-DE1A-DE11-97EE-003048724749.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/46ECF224-E81A-DE11-99A0-0030487D7B79.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/8AC19B36-EC1A-DE11-99EA-0030487E4B8D.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/B4899F1C-E81A-DE11-92FF-0030487F1797.root',
    '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V11_redigi_v1/0005/E6298150-E71A-DE11-A7FD-0030487F92A5.root'    
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")

# produce PAT Layer 1
process.load("PhysicsTools.PatAlgos.patSequences_cff")
# switch old trigger matching off
from PhysicsTools.PatAlgos.tools.trigTools import switchOffTriggerMatchingOld
switchOffTriggerMatchingOld( process )

process.TFileService=cms.Service("TFileService",
    fileName=cms.string("analyzePatElectron.root")
)

# calcultae the efficiency for electron reconstruction
# from the simulation
process.load("PhysicsTools.PatExamples.PatElectronAnalyzer_cfi")

# calculate the efficiency for electronID from a tag
# and probe method
process.load("PhysicsTools.PatExamples.tagAndProbeAnalysis_cff")

process.p = cms.Path(
    process.patDefaultSequence *
    process.analyzePatElectron *
    process.tagAndProbeAnalysis
)
