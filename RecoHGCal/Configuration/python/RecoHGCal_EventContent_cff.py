import FWCore.ParameterSet.Config as cms


#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_*ticlMultiClustersFromTrackstersEM_*_*',
      'keep *_*ticlMultiClustersFromTrackstersHAD_*_*',
      'keep *_*ticlMultiClustersFromTrackstersTrk_*_*',
      'keep *_*ticlMultiClustersFromTrackstersMIP_*_*',
      'keep *_*ticlMultiClustersFromTrackstersMerge_*_*',
      )
    )

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_*ticlTrackstersEM_*_*',
      'keep *_*ticlTrackstersHAD_*_*',
      'keep *_*ticlTrackstersTrk_*_*',
      'keep *_*ticlTrackstersMIP_*_*',
      'keep *_*ticlTrackstersMerge_*_*',
      'keep *_*ticlCandidateFromTracksters_*_*',
      'keep *_pfTICL_*_*'
      )
    )
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)

