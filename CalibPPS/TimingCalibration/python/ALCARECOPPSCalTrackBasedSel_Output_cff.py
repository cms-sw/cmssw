import FWCore.ParameterSet.Config as cms

OutALCARECOPPSCalTrackBasedSel_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSCalTrackBasedSel')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOPPSCalTrackBasedSel_*_*',
        'keep *_ctppsDiamondRawToDigi_*_*',
        'keep *_ctppsDiamondRecHits_*_*',
        'keep *_totemTimingRawToDigi_*_*',
        'keep *_totemTimingRecHits_*_*',
        'keep *_ctppsLocalTrackLiteProducer_*_*'
    )
)

OutALCARECOPPSCalTrackBasedSel = OutALCARECOPPSCalTrackBasedSel_noDrop.clone()
OutALCARECOPPSCalTrackBasedSel.outputCommands.insert(0, 'drop *')
