import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge

trackstersIters = ['keep *_ticlTracksters'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]

#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      trackstersIters +
      ['keep *_ticlTrackstersHFNoseTrkEM_*_*',
       'keep *_ticlTrackstersHFNoseEM_*_*',
       'keep *_ticlTrackstersHFNoseTrk_*_*',
       'keep *_ticlTrackstersHFNoseMIP_*_*',
       'keep *_ticlTrackstersHFNoseHAD_*_*',
       'keep *_ticlTrackstersHFNoseMerge_*_*',] +
      ['keep *_pfTICL_*_*']
      )
    )
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)
