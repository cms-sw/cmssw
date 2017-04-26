############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import os
process = cms.Process("L1Tracklet")
 
 
############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')

process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')


############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
Source_Files = cms.untracked.vstring(
    "/store/relval/CMSSW_9_0_0_pre5/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v4_D11-v1/00000/6EAF64B8-29FE-E611-9AD8-0025905A48D8.root"
    )
process.source = cms.Source("PoolSource", fileNames = Source_Files)


############################################################
# Path definitions & schedule
############################################################

# beamspot 
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.BS = cms.Path(process.offlineBeamSpot)

# run cluster+stubs associators
process.TTClusterStubAssociator = cms.Path(process.TrackTriggerAssociatorClustersStubs)

#run the tracking
process.TTTracks = cms.EDProducer("L1TrackProducer",
				 SimTrackSource = cms.InputTag("g4SimHits"),
				 SimVertexSource = cms.InputTag("g4SimHits"),
				 TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
				 TTStubMCTruthSource = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
                 BeamSpotSource = cms.InputTag("offlineBeamSpot")
    )
process.TrackTriggerTTTracks = cms.Sequence(process.TTTracks)
process.TT_step = cms.Path(process.TrackTriggerTTTracks)
process.TTAssociator = cms.Path(process.TrackTriggerAssociatorTracks)


#output module
#process.out = cms.OutputModule( "PoolOutputModule",
#                                fileName = cms.untracked.string("Tracklets.root"),
#                                fastCloning = cms.untracked.bool( False ),
#                                outputCommands = cms.untracked.vstring('keep *')
#)
#process.FEVToutput_step = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.TTClusterStubAssociator,process.BS,process.TT_step,process.TTAssociator)

