import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

# Load standard configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# Set the global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_dataRun3_HLT_v3', '')

# Configure the MessageLogger
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2024I/ScoutingPFRun3/HLTSCOUT/v1/000/386/478/00000/0100d00a-69a6-4710-931f-b1c660f87675.root'  # Replace with your input file
    )
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

# ScoutingElectronBestTrack producer
process.run3ScoutingElectronBestTrack = cms.EDProducer('Run3ScoutingElectronBestTrackProducer',
    Run3ScoutingElectron = cms.InputTag('hltScoutingEgammaPacker'),
    TrackPtMin = cms.vdouble(12.0, 12.0),
    TrackChi2OverNdofMax = cms.vdouble(3.0, 2.0),
    RelativeEnergyDifferenceMax = cms.vdouble(1.0, 1.0),
    DeltaPhiMax = cms.vdouble(0.06, 0.06)
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('output_file.root'),  # Replace with your output file
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_run3ScoutingElectronBestTrack_*_*')
)

# Path and EndPath definitions
process.p = cms.Path(process.run3ScoutingElectronBestTrack)
process.e = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.p, process.e)
