import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(==NUMEVT==)
)

# DiPions in energy bins

process.source = cms.Source("EmptySource")
process.generator = cms.EDProducer(
    "FlatRandomPtGunProducer",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinPt = cms.double(0.0),
        MaxPt = cms.double(1.0),
        MinEta = cms.double(-2.8),
        MaxEta = cms.double(+2.8),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
    ),
    AddAntiParticle = cms.bool(False), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)
)    

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

process.RandomNumberGeneratorService.theSource.initialSeed= ==seed1==
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

process.load("Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.MessageLogger = cms.Service("MessageLogger",
    reco = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
   destinations = cms.untracked.vstring('reco')
)

##from Kevin
process.firstStepHighPurity = cms.EDFilter("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("firstStepTracksWithQuality")
                                           )
process.fevt = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("fevt.root"),
    outputCommands = cms.untracked.vstring(
      'drop *',
      ###---these are the collection used in input to the "general tracks"
      # zero step 
      'keep *_zeroStep*_*_*',
      # step one
      'keep *_preMergingFirstStepTracksWithQuality*_*_*',
      # first(merged 0+1)  iterative step
      'keep *_firstStep*_*_*',
      # second step high quality
      'keep *_secStep_*_*',
      #third step  high quality
      'keep *_thStep_*_*',
      #fourth step high quality
      'keep *_pixellessStep*_*_*',
      #fifth step high quality
      'keep *_tobtecStep*_*_*',
      # merge of secStep+thStep 
      'keep *_merge2nd3rdTracks*_*_*',
      # merge of merge2nd3rd+pixelless
      'keep *_iterTracks*_*_*',
      # merge of pixellessStep+tobtecStep 
      'keep *_merge4th5thTracks*_*_*',
      #merge of firstStepTracksWithQuality+iterTracks
      "keep *_generalTracks_*_*",      
      'keep *_*Seed*_*_*',
      'keep *_sec*_*_*',
      'keep *_th*_*_*',
      'keep *_fou*_*_*',
      'keep *_fifth*_*_*',
      'keep *_newTrackCandidateMaker_*_*',
      "keep SimTracks_*_*_*",
      "keep SimVertexs_*_*_*",
      "keep edmHepMCProduct_*_*_*"
      )
)

process.p0 = cms.Path(process.generator+process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p3 = cms.Path(process.L1Emulator)
process.p4 = cms.Path(process.DigiToRaw)
process.p5= cms.Path(process.RawToDigi)
process.p6= cms.Path(process.reconstruction+
  #select only High purity step 1 fullsim tracks
                     process.firstStepHighPurity)
process.outpath = cms.EndPath(process.fevt)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p4,process.p5,process.p6,process.outpath)


