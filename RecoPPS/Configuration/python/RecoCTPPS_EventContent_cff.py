import FWCore.ParameterSet.Config as cms

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

    # totem T2
    'keep TotemFEDInfos_totemT2Digis_*_*',
    'keep TotemT2DigiedmNewDetSetVector_totemT2Digis_*_*',
    'keep TotemVFATStatusedmDetSetVector_totemT2Digis_*_*',
    'keep TotemT2RecHitedmNewDetSetVector_totemT2RecHits_*_*',

    # timing diamonds
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*',
    'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*',
    
    #diamond sampic
    'keep TotemTimingLocalTrackedmDetSetVector_diamondSampicLocalTracks_*_*',

    # TOTEM timing
    'keep TotemTimingDigiedmDetSetVector_totemTimingRawToDigi_*_*',
    'keep TotemTimingRecHitedmDetSetVector_totemTimingRecHits_*_*',
    'keep TotemTimingLocalTrackedmDetSetVector_totemTimingLocalTracks_*_*',

    # tracking pixels
    'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*',
    'keep CTPPSPixelDataErroredmDetSetVector_ctppsPixelDigis_*_*',
    'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',
    'keep CTPPSPixelRecHitedmDetSetVector_ctppsPixelRecHits_*_*',
    'keep CTPPSPixelLocalTrackedmDetSetVector_ctppsPixelLocalTracks_*_*',

    # CTPPS common
    'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*',
    'keep recoForwardProtons_ctppsProtons_*_*',
  )
)


RecoCTPPSRECO = cms.PSet(
  outputCommands = cms.untracked.vstring()
)


RecoCTPPSFEVT = cms.PSet(
  outputCommands = cms.untracked.vstring()
)

RecoCTPPSRECO.outputCommands.extend(RecoCTPPSAOD.outputCommands)
RecoCTPPSFEVT.outputCommands.extend(RecoCTPPSRECO.outputCommands)
