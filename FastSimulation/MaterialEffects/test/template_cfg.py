
import FWCore.ParameterSet.Config as cms

process = cms.Process("T")

# Number of events to be generated 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(80000)
)

# Include the RandomNumberGeneratorService definition
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(==SEED==),
        engineName = cms.untracked.string('TRandom3')
    ),
    # This is to initialize the random engine of the source
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(==SEED==),
        engineName = cms.untracked.string('TRandom3')
    ),
    # This is to initialize the random engines used for  Famos
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(==SEED==),
        engineName = cms.untracked.string('TRandom3')
    )
)

# Flat energy gun
process.source = cms.Source(
    "FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(==PID==),
        MinEta = cms.untracked.double(-5.10000),
        MaxEta = cms.untracked.double(5.10000),
        MinPhi = cms.untracked.double(-3.14159), ## in radians
        MaxPhi = cms.untracked.double(3.14159),
        MinE = cms.untracked.double(==ENERGY==),
        MaxE = cms.untracked.double(==ENERGY==),
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts
)

# Flat pT gun
"""
process.source = cms.Source(
    "FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(==PID==),
        MinEta = cms.untracked.double(-5.10000),
        MaxEta = cms.untracked.double(5.10000),
        MinPhi = cms.untracked.double(-3.14159), ## in radians
        MaxPhi = cms.untracked.double(3.14159),
        MinPt = cms.untracked.double(==PT==),
        MaxPt = cms.untracked.double(==PT==),
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts
)
"""

# Geant4-based CMS Detector simulation (OscarProducer)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Geometry.CMSCommonData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Configuration.StandardSequences.VtxSmearedNoSmear_cff")
process.psim = cms.Sequence(
    process.VtxSmeared+
    process.g4SimHits
)

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False)
)

process.outSim = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*', 
        'keep SimTracks_*_*_*', 
        'keep SimVertexs_*_*_*'),
    fileName = cms.untracked.string('==OUTPUT==')
)

process.p1 = cms.Path(process.psim)
process.outpath = cms.EndPath(process.outSim)

# Overall schedule
process.schedule = cms.Schedule(process.p1,process.outpath)



