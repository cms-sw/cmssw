import FWCore.ParameterSet.Config as cms

process = cms.Process("DigitizationReconstruction")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoPixelVertexing.PixelLowPtUtilities.MinBiasTracking_cff")

###############################################################################
# Categories and modules
process.CategoriesAndModules = cms.PSet(
  categories = cms.untracked.vstring('MinBiasTracking', 'NewVertices'),
  debugModules = cms.untracked.vstring('*')
)

###############################################################################
# Message logger
process.MessageLogger = cms.Service("MessageLogger",
    process.CategoriesAndModules,
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
#       threshold = cms.untracked.string('ERROR'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cerr')
)

###############################################################################
# Source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames  = cms.untracked.vstring(
       # /RelValMinBias/CMSSW_3_1_2-MC_31X_V3-v1/GEN-SIM-DIGI-RAW-HLTDEBUG
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0007/A0755F1D-9278-DE11-A9F7-001D09F25208.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/D01E77F1-6378-DE11-8A4F-001D09F24F1F.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/B01EB0F1-6378-DE11-821E-001D09F28F11.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/AA7E66EB-6378-DE11-99B2-0019B9F70607.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/3CC6F3F2-6378-DE11-B308-001D09F28E80.root'
       # /RelValMinBias/CMSSW_3_1_2-STARTUP31X_V2-v1/GEN-SIM-DIGI-RAW-HLTDEBUG
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/7C2724F8-9178-DE11-AFCE-001D09F244BB.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/C4012D09-7978-DE11-B872-001D09F28F1B.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/A2F7FF3A-7878-DE11-BD8B-001D09F26C5C.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/68F0C546-7E78-DE11-BEFF-0019B9F709A4.root',
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/687383FC-7D78-DE11-BD74-001D09F34488.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

###############################################################################
# Energy loss
process.energyLoss = cms.EDProducer("EnergyLossProducer",
    pixelToStripMultiplier = cms.double(1.23),
    pixelToStripExponent   = cms.double(1.0),
    trackProducer          = cms.string('allTracks')
)

###############################################################################
# Track associator
process.TrackAssociatorByHits.SimToRecoDenominator = 'reco'
process.TrackAssociatorByHits.Quality_SimToReco = cms.double(0.5)
process.TrackAssociatorByHits.Purity_SimToReco  = cms.double(0.5)
process.TrackAssociatorByHits.Cut_RecoToSim     = cms.double(0.5)

process.TrackAssociatorByHits.associatePixel    = cms.bool(True)
process.TrackAssociatorByHits.associateStrip    = cms.bool(False)

###############################################################################
# Track analyzer
process.analyzeTracks = cms.EDAnalyzer("HadronAnalyzer",
    hasSimInfo             = cms.bool(True),
    trackProducer          = cms.string('allTracks'),
    allRecTracksArePrimary = cms.bool(False),
    fillHistograms         = cms.bool(True),
    histoFile              = cms.string('histograms.root'),
    fillNtuples            = cms.bool(False),
    ntupleFile             = cms.string('ntuples.root')
)

###############################################################################
# Event plotter
process.plotEvent = cms.EDAnalyzer("EventPlotter",
    hasSimInfo             = cms.bool(True),
    trackProducer          = cms.string('allTracks')
)

###############################################################################
# Paths
process.simu  = cms.Path(process.mix
                       * process.trackingParticles
                       * process.offlineBeamSpot)

process.digi  = cms.Path(process.RawToDigi)

process.lreco = cms.Path(process.trackerlocalreco)

process.greco = cms.Path(process.minBiasTracking
                       * process.energyLoss
                       * process.pixelVZeros
                       * process.analyzeTracks)

###############################################################################
# Global tag
process.GlobalTag.globaltag = 'MC_31X_V3::All'

###############################################################################
# Schedule
process.schedule = cms.Schedule(process.simu,
                                process.digi,
                                process.lreco,
                                process.greco)
