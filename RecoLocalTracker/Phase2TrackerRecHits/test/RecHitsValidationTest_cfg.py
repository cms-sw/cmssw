# Imports
import FWCore.ParameterSet.Config as cms

# Create a new CMS process
process = cms.Process('cluTest')

# Import all the necessary files
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:step3.root')
)

# Output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:rechits_validation.root')
)

process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi')
#process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEGeometricESProducer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi')
#process.siPhase2RecHits.Phase2StripCPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE")
#process.siPhase2RecHits.Phase2StripCPE = cms.ESInputTag("phase2StripCPEGeometricESProducer", "Phase2StripCPEGeometric")


# Analyzer
process.analysis = cms.EDAnalyzer('Phase2TrackerRecHitsValidation',
    src = cms.InputTag("siPhase2RecHits"),
    clusters = cms.InputTag("siPhase2Clusters"),
    links = cms.InputTag("simSiPixelDigis", "Tracker"),
    simhitsbarrel = cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
    simhitsendcap = cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
    simtracks = cms.InputTag("g4SimHits"),
    ECasRings = cms.bool(True),
    SimTrackMinPt = cms.double(2.),
    MakeEtaPlots = cms.bool(False),
    MinEta = cms.double(0.),
    MaxEta = cms.double(10.)
)

# Processes to run
#process.rechits_step = cms.Path(process.siPhase2Clusters + process.siPhase2RecHits)
process.rechits_step = cms.Path(process.siPhase2RecHits)
process.validation_step = cms.Path(process.analysis)

process.schedule = cms.Schedule(process.rechits_step, process.validation_step)

