import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelRecHitsInputDistributionsMaking")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.FakeConditions_cff")

# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
# include "Configuration/StandardSequences/data/MagneticField_38T.cff"
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

process.load("SimG4Core.Application.g4SimHits_cfi")

# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# TRACKER digitization sequence
process.load("SimTracker.Configuration.SimTracker_cff")

# TRACKER LocalReco sequence 
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

# TRACKER rechits validation sequence
process.load("FastSimulation.TrackingRecHitProducer.test.SiPixelRecHitsInputDistributionsMaker_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        siStripDigis = cms.untracked.uint32(7654),
        siPixelDigis = cms.untracked.uint32(8765),
        mix = cms.untracked.uint32(7697)
    ),
    # Change this seed to produce more samples
    sourceSeed = cms.untracked.uint32(1)
)

process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(100.01),
        MinPt = cms.untracked.double(99.99),
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        #
        # phi must be given in radians
        #
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    AddAntiParticle = cms.untracked.bool(True), ## if you turn it ON, for each particle

    firstRun = cms.untracked.uint32(1)
)

process.USER = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pixelrechits.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits)
process.digis = cms.Sequence(process.trDigi)
process.rechits = cms.Sequence(process.trackerlocalreco*process.pixRecHitsDistributionsMaker)
process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.rechits)
process.outpath = cms.EndPath(process.USER)
process.g4SimHits.Generator.HepMCProductLabel = 'source'


