import FWCore.ParameterSet.Config as cms

RecoCTPPSFEVT = cms.PSet(
  outputCommands = cms.untracked.vstring(
    # trigger counters
    'keep TotemTriggerCounters_totemTriggerRawToDigi_*_*',

    # tracking strip
    'keep TotemFEDInfos_totemRPRawToDigi_*_*',
    'keep TotemRPDigiedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemRPClusteredmDetSetVector_totemRPClusterProducer_*_*',
    'keep TotemRPRecHitedmDetSetVector_totemRPRecHitProducer_*_*',
    'keep TotemRPUVPatternedmDetSetVector_totemRPUVPatternFinder_*_*',
    'keep TotemRPLocalTrackedmDetSetVector_totemRPLocalTrackFitter_*_*',

    # timing diamonds
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*',
    'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*',

    #tracking pixels
    'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*',
    'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',

    # CTPPS common
    'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*'
  )
)


RecoCTPPSRECO = cms.PSet(
  outputCommands = cms.untracked.vstring(
    # trigger counters
    'keep TotemTriggerCounters_totemTriggerRawToDigi_*_*',

    # tracking strip
    'keep TotemFEDInfos_totemRPRawToDigi_*_*',
    'keep TotemRPDigiedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemRPClusteredmDetSetVector_totemRPClusterProducer_*_*',
    'keep TotemRPRecHitedmDetSetVector_totemRPRecHitProducer_*_*',
    'keep TotemRPUVPatternedmDetSetVector_totemRPUVPatternFinder_*_*',
    'keep TotemRPLocalTrackedmDetSetVector_totemRPLocalTrackFitter_*_*',

    #tracking pixels
    'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*',
    'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',

    # timing diamonds
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*',
    'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*',

    # CTPPS common
    'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*'
  )
)


RecoCTPPSAOD = cms.PSet(
  outputCommands = cms.untracked.vstring(
    # trigger counters
    'keep TotemTriggerCounters_totemTriggerRawToDigi_*_*',

    # tracking strip
    'keep TotemFEDInfos_totemRPRawToDigi_*_*',
    'keep TotemRPDigiedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_totemRPRawToDigi_*_*',
    'keep TotemRPClusteredmDetSetVector_totemRPClusterProducer_*_*',
    'keep TotemRPRecHitedmDetSetVector_totemRPRecHitProducer_*_*',
    'keep TotemRPUVPatternedmDetSetVector_totemRPUVPatternFinder_*_*',
    'keep TotemRPLocalTrackedmDetSetVector_totemRPLocalTrackFitter_*_*',

    # timing diamonds
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*',
    'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*',

    #tracking pixels
    'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*',
    'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',

    # CTPPS common
    'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*'
  )
)
