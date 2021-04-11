# define basic process
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
process = cms.Process("L1HybridTrack")

# ----------------------------------------------------------------------------------
# import standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

############################################################
# L1 tracking: remake stubs?
############################################################
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')


# ----------------------------------------------------------------------------------
# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
Source_Files = cms.untracked.vstring(
    "/store/relval/CMSSW_11_2_0_pre5/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/FDFA00CE-FA93-0142-B187-99CBD4A43944.root"
)
process.source = cms.Source("PoolSource", fileNames = Source_Files)

# L1 tracking => hybrid emulation 
process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")

# prompt tracking only
process.TTTracksEmulation = cms.Path(process.L1HybridTracks)
process.TTTracksEmulationWithTruth = cms.Path(process.L1HybridTracksWithAssociators)

# extended tracking only
#process.TTTracksEmulation = cms.Path(process.L1ExtendedHybridTracks)
#process.TTTracksEmulationWithTruth = cms.Path(process.L1ExtendedHybridTracksWithAssociators)

# both prompt+extended hybrid tracking
#process.TTTracksEmulation = cms.Path(process.L1PromptExtendedHybridTracks)
#process.TTTracksEmulationWithTruth = cms.Path(process.L1PromptExtendedHybridTracksWithAssociators)

# -----------------------------------------------------------------------------------
# DTC Emulations
process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
process.load( 'L1Trigger.TrackerDTC.ProducerES_cff' )

#--- Load code that produces DTCStubs

# load Track Trigger Configuration
process.load( 'L1Trigger.TrackerDTC.ProducerES_cff' )
# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
# load code that analyzes DTCStubs
process.load( 'L1Trigger.TrackerDTC.Analyzer_cff' )

process.dtc = cms.Path( process.TrackerDTCProducer )



# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# output module
process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string("L1Tracks.root"),
                                fastCloning = cms.untracked.bool( False ),
                                outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_TTTrack*_Level1TTTracks_*', 
)
)
process.FEVToutput_step = cms.EndPath(process.out)

  
#process.schedule = cms.Schedule(process.TTTracksEmulation,process.FEVToutput_step)
process.schedule = cms.Schedule(process.dtc,process.TTTracksEmulationWithTruth,process.FEVToutput_step)

