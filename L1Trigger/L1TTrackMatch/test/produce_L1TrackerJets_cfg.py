############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TrackJets")
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff') ## this needs to match the geometry you are running on
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')     ## this needs to match the geometry you are running on

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')



from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

############################################################
# input and output
############################################################


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
Source_Files = cms.untracked.vstring('file:/mnt/hadoop/store/user/rish/MinBiasPU200/reprocessMinBias10.root')
#process.load("L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff")
#process.TTTracks = cms.Path(process.L1TrackletTracks)                         #run only the tracking (no MC truth associators)
#process.TTTracksWithTruth = cms.Path(process.L1TrackletTracksWithAssociators)


process.source = cms.Source("PoolSource", fileNames = Source_Files) #, inputCommands=cms.untracked.vstring('drop *EMTF_*_*_*'))
process.load("L1Trigger.L1TTrackMatch.L1TkPrimaryVertexProducer_cfi")
process.pL1TkPrimaryVertex = cms.Path( process.L1TkPrimaryVertex )
process.load("L1Trigger.L1TTrackMatch.L1TrackerJetProducer_cfi");
process.pL1TrackerJets=cms.Path(process.L1TrackerJets)
process.out = cms.OutputModule( "PoolOutputModule",
                                fastCloning = cms.untracked.bool( False ),
                                fileName = cms.untracked.string("test.root" )
		               )
process.FEVToutput_step = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.TTTracks,process.TTTracksWithTruth,process.pL1TkPrimaryVertex,process.pL1TrackerJets,process.FEVToutput_step)
