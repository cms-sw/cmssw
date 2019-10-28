import FWCore.ParameterSet.Config as cms


# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_tracksters*_*_*',
      'keep *_multiClustersFromTracksters*_*_*',
      'keep *_ticlCandidateFromTrackstersProducer*_*_*',
      'keep *_pfTICLProducer*_*_*'
      )
    )
#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_tracksters*_*_*',
      'keep *_multiClustersFromTracksters*_*_*',
      'keep *_ticlCandidateFromTrackstersProducer*_*_*',
      'keep *_pfTICLProducer*_*_*'
      )
    )

#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_multiClustersFromTracksters*_*_*'
      )
    )

