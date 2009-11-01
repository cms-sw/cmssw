import FWCore.ParameterSet.Config as cms

process = cms.Process("DigitizationReconstruction")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoPixelVertexing.PixelLowPtUtilities.MinBiasTracking_cff")

###############################################################################
# Categories and modules
process.CategoriesAndModules = cms.PSet(
    categories = cms.untracked.vstring('MinBiasTracking'),
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
    # FIXME
    skipEvents = cms.untracked.uint32(0),
    fileNames  = cms.untracked.vstring(
       # RelValMinBias/CMSSW_3_1_0_pre5_IDEAL_31X_v1/GEN-SIM-RECO
       '/store/relval/CMSSW_3_1_0_pre5/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0000/3C9D7BDE-B62B-DE11-A7FA-000423D94524.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0000/707E2511-B62B-DE11-B0B0-001D09F24498.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0000/CE506207-0C2C-DE11-AC5E-000423D991F0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

###############################################################################
# Energy loss
process.energyLoss = cms.EDProducer("EnergyLossProducer",
    pixelToStripMultiplier = cms.double(1.23),
    pixelToStripExponent   = cms.double(1.0),
    trackProducer          = cms.string('allTracks')
)

###############################################################################
# Track analyzer
process.TrackAssociatorByHits.SimToRecoDenominator = 'reco'

process.analyzeTracks = cms.EDAnalyzer("QCDTrackAnalyzer",
    allRecTracksArePrimary = cms.bool(False),
    hasSimInfo             = cms.bool(True),
    trackProducer          = cms.string('allTracks'),
    fillHistograms         = cms.bool(True),
    histoFile              = cms.string('histograms.root'),
    fillNtuples            = cms.bool(False),
    ntupleFile             = cms.string('ntuples.root')
)

###############################################################################
# Event plotter
process.plotEvent = cms.EDAnalyzer("EventPlotter",
    trackProducer = cms.string('allTracks')
)

###############################################################################
# Paths
process.r2d   = cms.Path(process.RawToDigi)

process.simu  = cms.Path(process.mix
                       * process.trackingParticles
                       * process.offlineBeamSpot)

process.lreco = cms.Path(process.trackerlocalreco
                       * process.ecalLocalRecoSequence)

process.greco = cms.Path(process.minBiasTracking
                       * process.energyLoss
                       * process.pixelVZeros
                       * process.analyzeTracks)
#                      * process.plotEvent)

###############################################################################
# Global tag
process.GlobalTag.globaltag = 'IDEAL_31X::All'

###############################################################################
# Schedule
process.schedule = cms.Schedule(process.r2d,
                                process.simu,
                                process.lreco,
                                process.greco)

