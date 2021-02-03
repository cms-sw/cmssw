import FWCore.ParameterSet.Config as cms

hcalRecAlgos = cms.ESProducer("HcalRecAlgoESProducer",
    DropChannelStatusBits = cms.vstring(
        'HcalCellMask',
        'HcalCellOff',
        'HcalCellDead'
    ),
    RecoveredRecHitBits = cms.vstring(''),
    SeverityLevels = cms.VPSet(
        cms.PSet(
            ChannelStatus = cms.vstring(''),
            Level = cms.int32(0),
            RecHitFlags = cms.vstring('')
        ),
        cms.PSet(
            ChannelStatus = cms.vstring('HcalCellCaloTowerProb'),
            Level = cms.int32(1),
            RecHitFlags = cms.vstring('')
        ),
        cms.PSet(
            ChannelStatus = cms.vstring('HcalCellExcludeFromHBHENoiseSummary'),
            Level = cms.int32(5),
            RecHitFlags = cms.vstring(
                'HBHEIsolatedNoise',
                'HFAnomalousHit'
            )
        ),
        cms.PSet(
            ChannelStatus = cms.vstring(''),
            Level = cms.int32(8),
            RecHitFlags = cms.vstring(
                'HBHEHpdHitMultiplicity',
                'HBHEFlatNoise',
                'HBHESpikeNoise',
                'HBHETS4TS5Noise',
                'HBHENegativeNoise',
                'HBHEOOTPU'
            )
        ),
        cms.PSet(
            ChannelStatus = cms.vstring(''),
            Level = cms.int32(11),
            RecHitFlags = cms.vstring(
                'HFLongShort',
                'HFS8S1Ratio',
                'HFPET',
                'HFSignalAsymmetry'
            )
        ),
        cms.PSet(
            ChannelStatus = cms.vstring('HcalCellCaloTowerMask'),
            Level = cms.int32(12),
            RecHitFlags = cms.vstring('')
        ),
        cms.PSet(
            ChannelStatus = cms.vstring('HcalCellHot'),
            Level = cms.int32(15),
            RecHitFlags = cms.vstring('')
        ),
        cms.PSet(
            ChannelStatus = cms.vstring(
                'HcalCellOff',
                'HcalCellDead'
            ),
            Level = cms.int32(20),
            RecHitFlags = cms.vstring('')
        )
    ),
    appendToDataLabel = cms.string(''),
    phase = cms.uint32(1)
)
