
import FWCore.ParameterSet.Config as cms

process = cms.Process("T")

# Number of events to be generated 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Include the RandomNumberGeneratorService definition
#
process.load("Configuration.StandardSequences.Services_cff")

#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CMSCommonData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_31X_V8::All"

process.load("FWCore.MessageService.MessageLogger_cfi")

# this config frament brings you the generator information
process.load("Configuration.StandardSequences.Generator_cff")

# this config frament brings you 3 steps of the detector simulation:
# -- vertex smearing (IR modeling)
# -- G4-based hit level detector simulation
# -- digitization (electronics readout modeling)
# it returns 2 sequences : 
# -- psim (vtx smearing + G4 sim)
# -- pdigi (digitization in all subsystems, i.e. tracker=pix+sistrips,
#           cal=ecal+ecal-0-suppression+hcal), muon=csc+dt+rpc)
#
process.load("Configuration.StandardSequences.Simulation_cff")

process.RandomNumberGeneratorService.generator.initialSeed= ==SEED==

# please note the IMPORTANT: 
# in order to operate Digis, one needs to include Mixing module 
# (pileup modeling), at least in the 0-pileup mode
#
# There're 3 possible configurations of the Mixing module :
# no-pileup, low luminosity pileup, and high luminosity pileup
#
# they come, respectively, through the 3 config fragments below
#
# *each* config returns label "mix"; thus you canNOT have them
# all together in the same configuration, but only one !!!
#
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

#include "Configuration/StandardSequences/data/MixingLowLumiPileUp.cff" 
#include "Configuration/StandardSequences/data/MixingHighLumiPileUp.cff" 
process.load("Configuration.StandardSequences.L1Emulator_cff")

process.load("Configuration.StandardSequences.DigiToRaw_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.VtxSmearedNoSmear_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.RandomNumberGeneratorService.theSource.initialSeed= ==SEED==

process.source = cms.Source("EmptySource")

# Flat energy gun
"""
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
"""
# Flat pT gun
process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(==PID==),
        # you can request more than 1 particle
        # PartID = cms.vint32(211,11,-13),
        MinPt = cms.double(==PT==),
        MaxPt = cms.double(==PT==),
        MinEta = cms.double(-5.1),
        MaxEta = cms.double(5.1),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
    ),
    AddAntiParticle = cms.bool(False), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)   
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

process.MessageLogger = cms.Service("MessageLogger",
    reco = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
   destinations = cms.untracked.vstring('reco')
)

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

process.p1 = cms.Path(process.generator+process.pgen+process.psim)
process.outpath = cms.EndPath(process.outSim)

# Overall schedule
process.schedule = cms.Schedule(process.p1,process.outpath)



