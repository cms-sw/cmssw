############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import os
process = cms.Process("L1Tracklet")
 
GEOMETRY = "D17"


############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

if GEOMETRY == "D17":
    print "using geometry " + GEOMETRY + " (tilted)"
    process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
else:
    print "this is not a valid geometry!!!"

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

if GEOMETRY == "D17":
    Source_Files = cms.untracked.vstring(
        "/store/relval/CMSSW_9_3_2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/10000/0681719F-AFA6-E711-87C9-0CC47A4C8E14.root"
    )
process.source = cms.Source("PoolSource", fileNames = Source_Files)


############################################################
# remake L1 stubs and/or cluster/stub truth ??
############################################################

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")

#if GEOMETRY == "D10": 
#    TTStubAlgorithm_official_Phase2TrackerDigi_.zMatchingPS = cms.bool(False)

process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs)


############################################################
# L1 tracking
############################################################

#from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *
#if GEOMETRY == "D10": 
#    TTTracksFromTracklet.trackerGeometry = cms.untracked.string("flat")
#TTTracksFromTracklet.asciiFileName = cms.untracked.string("evlist.txt")

process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
process.TTTracks = cms.Path(process.L1TrackletTracks)
process.TTTracksWithTruth = cms.Path(process.L1TrackletTracksWithAssociators)


############################################################
# output module
############################################################

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("Tracklet_L1Tracks.root"),
                               fastCloning = cms.untracked.bool(False),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_TTTrack*_Level1TTTracks_*', 
#                                                                      'keep *_TTCluster*_*_*',
#                                                                      'keep *_TTStub*_*_*'
)
)
process.FEVToutput_step = cms.EndPath(process.out)


# use this if you want to re-run the stub making
#process.schedule = cms.Schedule(process.TTClusterStub,TrackTriggerAssociatorClustersStubs,process.TTTracksWithTruth,process.FEVToutput_step)

# use this if cluster/stub associators not available 
process.schedule = cms.Schedule(process.TTClusterStubTruth,process.TTTracksWithTruth,process.FEVToutput_step)

# use this to only run tracking + track associator
#process.schedule = cms.Schedule(process.TTTracksWithTruth,process.FEVToutput_step)

