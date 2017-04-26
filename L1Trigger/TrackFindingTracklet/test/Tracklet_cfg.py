# define basic process
import FWCore.ParameterSet.Config as cms
import os
process = cms.Process("L1Tracklet")
 

# import standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
Source_Files = cms.untracked.vstring(
    "/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/6EAF64B8-29FE-E611-9AD8-0025905A48D8.root"
    )
process.source = cms.Source("PoolSource", fileNames = Source_Files)


# L1 tracking
process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
process.TTTracks = cms.Path(process.L1TrackletTracks)                         #run only the tracking (no MC truth associators)
process.TTTracksWithTruth = cms.Path(process.L1TrackletTracksWithAssociators) #run the tracking AND MC truth associators)


# output module
process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string("Tracklets.root"),
                                fastCloning = cms.untracked.bool( False ),
                                outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_TTTrack*_Level1TTTracks_*', 
#                                                                       'keep *_TTCluster*_*_*',
#                                                                       'keep *_TTStub*_*_*'
)
)
process.FEVToutput_step = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.TTTracksWithTruth,process.FEVToutput_step)

