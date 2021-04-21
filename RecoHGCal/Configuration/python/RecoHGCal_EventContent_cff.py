import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge

trackstersIters = ['keep *_ticlTracksters'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]
trackstersHFNoseIters = ['keep *_ticlTrackstersHFNose'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      trackstersIters +
      trackstersHFNoseIters +
      ['keep *_pfTICL_*_*']
      )
    )

# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)
