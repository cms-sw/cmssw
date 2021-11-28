import FWCore.ParameterSet.Config as cms

OutALCARECOPPSCalMaxTracks_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSCalMaxTracks')
    ),
    outputCommands = cms.untracked.vstring(
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

        # tracking pixels
        'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*',
        'keep CTPPSPixelDataErroredmDetSetVector_ctppsPixelDigis_*_*',
        'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',
        'keep CTPPSPixelRecHitedmDetSetVector_ctppsPixelRecHits_*_*',
        'keep CTPPSPixelLocalTrackedmDetSetVector_ctppsPixelLocalTracks_*_*',

        # CTPPS common
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*',
        'keep recoForwardProtons_ctppsProtons_*_*',

        # HLT info
        'keep *_hltGtStage2ObjectMap_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*'     
    )
)

OutALCARECOPPSCalMaxTracks = OutALCARECOPPSCalMaxTracks_noDrop.clone()
OutALCARECOPPSCalMaxTracks.outputCommands.insert(0, 'drop *')
