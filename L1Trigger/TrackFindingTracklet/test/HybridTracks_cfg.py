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

process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff') 
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')

# ----------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
inputMC = ["/store/mc/CMSSW_12_6_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_125X_mcRun4_realistic_v5_2026D88PU200RV183v2-v1/30000/0959f326-3f52-48d8-9fcf-65fc41de4e27.root"]
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

# ----------------------------------------------------------------------------------
# DTC emulation
# ----------------------------------------------------------------------------------

process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
process.dtc = cms.Path( process.TrackerDTCProducer )

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
