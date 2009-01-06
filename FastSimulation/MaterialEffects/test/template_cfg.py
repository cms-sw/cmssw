
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
process.GlobalTag.globaltag = "IDEAL_30X::All"

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

process.RandomNumberGeneratorService.theSource.initialSeed= ==SEED==
#process.RandomNumberGeneratorService.theSource.initialSeed= 1414

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

process.MessageLogger = cms.Service("MessageLogger",
    reco = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
   destinations = cms.untracked.vstring('reco')
)

# Geant4-based CMS Detector simulation (OscarProducer)
#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimG4Core.Application.g4SimHits_cfi")
process.psim = cms.Sequence(
    process.VtxSmeared+
    process.g4SimHits
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

process.p1 = cms.Path(process.psim)
process.outpath = cms.EndPath(process.outSim)

# Overall schedule
process.schedule = cms.Schedule(process.p1,process.outpath)



