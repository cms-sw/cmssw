import FWCore.ParameterSet.Config as cms

OutALCARECOPPSCalMaxTracks_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSCalMaxTracks')
    ),
    outputCommands = cms.untracked.vstring(
        # timing diamonds
        'keep TotemFEDInfos_ctppsDiamondRawToDigiAlCaRecoProducer_*_*',
        'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigiAlCaRecoProducer_*_*',
        'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigiAlCaRecoProducer_*_*',
        'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHitsAlCaRecoProducer_*_*',
        'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracksAlCaRecoProducer_*_*',
        
        #diamond sampic
        'keep TotemTimingDigiedmDetSetVector_totemTimingRawToDigiAlCaRecoProducer_*_*',
        'keep TotemTimingRecHitedmDetSetVector_totemTimingRecHitsAlCaRecoProducer_*_*',
        'keep TotemTimingLocalTrackedmDetSetVector_diamondSampicLocalTracksAlCaRecoProducer_*_*',        

        # tracking pixels
        'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigisAlCaRecoProducer_*_*',
        'keep CTPPSPixelDataErroredmDetSetVector_ctppsPixelDigisAlCaRecoProducer_*_*',
        'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClustersAlCaRecoProducer_*_*',
        'keep CTPPSPixelRecHitedmDetSetVector_ctppsPixelRecHitsAlCaRecoProducer_*_*',
        'keep CTPPSPixelLocalTrackedmDetSetVector_ctppsPixelLocalTracksAlCaRecoProducer_*_*',

        # CTPPS common
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducerAlCaRecoProducer_*_*',
        'keep recoForwardProtons_ctppsProtonsAlCaRecoProducer_*_*',

        # HLT info
        'keep *_hltGtStage2ObjectMap_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*'          
    )
)

OutALCARECOPPSCalMaxTracks = OutALCARECOPPSCalMaxTracks_noDrop.clone()
OutALCARECOPPSCalMaxTracks.outputCommands.insert(0, 'drop *')