import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
# this example configuration offers some minimum 
# annotation, to help users get through; please
# don't hesitate to read through the comments
# use MessageLogger to redirect/suppress multiple
# service messages coming from the system
#
# in this config below, we use the replace option to make
# the logger let out messages of severity ERROR (INFO level
# will be suppressed), and we want to limit the number to 10
#
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

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

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        # since PartID is a vector, you can place in as many 
        # PDG id's as you wish, comma seaparated
        #
        PartID = cms.untracked.vint32(11),
        MaxEta = cms.untracked.double(2.7),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.7),
        MinE = cms.untracked.double(10.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## in radians

        MaxE = cms.untracked.double(10.0)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    AddAntiParticle = cms.untracked.bool(True) ## back-to-back particles

)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('PhysVal-DiElectron-Ene10.root')
)

process.p0 = cms.Path(process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p3 = cms.Path(process.L1Emulator)
process.p4 = cms.Path(process.DigiToRaw)
process.p5= cms.Path(process.RawToDigi)
process.p6= cms.Path(process.reconstruction)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p4,process.p5,process.p6,process.outpath)


