# ----------------------------------------------------------------------------------
# define basic process
# ----------------------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
process = cms.Process("L1HybridTrack")

# ----------------------------------------------------------------------------------
# import standard configurations
# ----------------------------------------------------------------------------------

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load( 'Configuration.Geometry.GeometryExtendedRun4D110Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtendedRun4D110_cff' )

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')

# ----------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )
options.parseArguments()
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )
inputMC = [""]
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC), skipEvents = cms.untracked.uint32( 17 ))

# ----------------------------------------------------------------------------------
# DTC emulation
# ----------------------------------------------------------------------------------

process.load( 'L1Trigger.TrackerDTC.DTC_cff' )
process.dtc = cms.Path( process.ProducerDTC )

# ----------------------------------------------------------------------------------
# L1 tracking
# ----------------------------------------------------------------------------------

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")

# prompt tracking only
process.TTTracksEmulation = cms.Path(process.L1THybridTracks)
process.TTTracksEmulationWithTruth = cms.Path(process.L1THybridTracksWithAssociators)

# extended tracking only
#process.TTTracksEmulation = cms.Path(process.L1TExtendedHybridTracks)
#process.TTTracksEmulationWithTruth = cms.Path(process.L1TExtendedHybridTracksWithAssociators)

# both prompt+extended hybrid tracking
#process.TTTracksEmulation = cms.Path(process.L1TPromptExtendedHybridTracks)
#process.TTTracksEmulationWithTruth = cms.Path(process.L1TPromptExtendedHybridTracksWithAssociators)

process.MessageLogger.L1track = dict(limit = -1)

# ----------------------------------------------------------------------------------
# output module
# ----------------------------------------------------------------------------------

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string("L1Tracks.root"),
                                fastCloning = cms.untracked.bool( False ),
                                outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_TTTrack*_Level1TTTracks_*', 
)
)
process.FEVToutput_step = cms.EndPath(process.out)

# ----------------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------------

process.schedule = cms.Schedule(process.dtc,process.TTTracksEmulationWithTruth,process.FEVToutput_step)
