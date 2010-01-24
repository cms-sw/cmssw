import FWCore.ParameterSet.Config as cms

process = cms.Process("ClusterShape")

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
#        threshold = cms.untracked.string('ERROR'),
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
       '/store/relval/CMSSW_3_1_2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/3CC6F3F2-6378-DE11-B308-001D09F28E80.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

###############################################################################
# Cluster shape
process.clusterShape = cms.EDAnalyzer("ClusterShapeExtractor",
    trackProducer  = cms.string('allTracks'),
    hasSimHits     = cms.bool(True),
    hasRecTracks   = cms.bool(False),
    associateStrip      = cms.bool(True),
    associatePixel      = cms.bool(True),
    associateRecoTracks = cms.bool(False),
    ROUList = cms.vstring(
      'g4SimHitsTrackerHitsTIBLowTof', 'g4SimHitsTrackerHitsTIBHighTof',
      'g4SimHitsTrackerHitsTIDLowTof', 'g4SimHitsTrackerHitsTIDHighTof',
      'g4SimHitsTrackerHitsTOBLowTof', 'g4SimHitsTrackerHitsTOBHighTof',
      'g4SimHitsTrackerHitsTECLowTof', 'g4SimHitsTrackerHitsTECHighTof',
      'g4SimHitsTrackerHitsPixelBarrelLowTof',
      'g4SimHitsTrackerHitsPixelBarrelHighTof',
      'g4SimHitsTrackerHitsPixelEndcapLowTof',
      'g4SimHitsTrackerHitsPixelEndcapHighTof')
)

###############################################################################
# Paths
process.simu  = cms.Path(process.mix
                       * process.trackingParticles
                       * process.offlineBeamSpot)

process.digi  = cms.Path(process.RawToDigi)

process.lreco = cms.Path(process.trackerlocalreco
                       * process.clusterShape)

###############################################################################
# Global tag
process.GlobalTag.globaltag = 'MC_31X_V3::All'

###############################################################################
# Schedule
process.schedule = cms.Schedule(process.simu,
                                process.digi,
                                process.lreco)

