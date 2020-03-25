import FWCore.ParameterSet.Config as cms


#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_multiClustersFromTrackstersEM_*_*',
      'keep *_multiClustersFromTrackstersHAD_*_*',
      'keep *_multiClustersFromTrackstersTrk_*_*',
      'keep *_multiClustersFromTrackstersMIP_*_*',
      'keep *_multiClustersFromTrackstersMerge_*_*',
      )
    )

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_trackstersEM_*_*',
      'keep *_trackstersHAD_*_*',
      'keep *_trackstersTrk_*_*',
      'keep *_trackstersMIP_*_*',
      'keep *_trackstersMerge_*_*',
      'keep *_ticlCandidateFromTrackstersProducer_*_*',
      'keep *_pfTICLProducer_*_*'
      )
    )
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)

