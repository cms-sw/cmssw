
import FWCore.ParameterSet.Config as cms

process = cms.Process("TIME")
process.load("HLTrigger.Timer.timer_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoHI.HiMuonAlgos.HiMuL3_cff")
process.load("RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout',
        'cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
)
process.Timing = cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    catalog = cms.untracked.string('PoolFileCatalog.xml'),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/pgun_upsilon2muons_d20080604/pgun_upsilon2muons_d20080604_r000001.root')
)

process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(False)
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.output = cms.OutputModule("PoolOutputModule",
# If you really want to filter on events:
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    ),

    fileName = cms.untracked.string('__OUTPUT__')
)
process.p1 = cms.Path(process.mix*process.doAllDigi*process.L1Emulator*process.DigiToRaw*process.RawToDigi*process.trackerlocalreco*process.muonlocalreco*process.offlineBeamSpot*process.MuonSeed*process.standAloneMuons*process.recopixelvertexing*process.muonFilter)

process.outpath = cms.EndPath(process.myTimer*process.output)

