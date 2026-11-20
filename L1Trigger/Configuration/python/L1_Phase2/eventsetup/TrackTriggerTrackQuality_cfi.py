import FWCore.ParameterSet.Config as cms

TrackTriggerTrackQuality = cms.ESProducer("trackerTFP::ProducerTrackQuality",
    BaseShiftAPfixed = cms.int32(-5),
    BaseShiftCot = cms.int32(-7),
    BaseShiftZ0 = cms.int32(-6),
    BaseShiftchi2rphi = cms.int32(-6),
    BaseShiftchi2rz = cms.int32(-6),
    Chi2rphiConv = cms.int32(3),
    Chi2rzConv = cms.int32(13),
    DphiTruncation = cms.int32(16),
    DzTruncation = cms.int32(262144),
    FeatureNames = cms.vstring(
        'tanl',
        'z0_scaled',
        'bendchi2_bin',
        'nstub',
        'nlaymiss_interior',
        'chi2rphi_bin',
        'chi2rz_bin'
    ),
    Model = cms.FileInPath('L1Trigger/TrackTrigger/data/L1_TrackQuality_GBDT_emulation_digitized.json'),
    WeightBinFraction = cms.int32(0),
    WidthInvV0 = cms.int32(16),
    WidthInvV1 = cms.int32(16),
    WidthM20 = cms.int32(16),
    WidthM21 = cms.int32(16),
    Widthchi2rphi = cms.int32(20),
    Widthchi2rz = cms.int32(20)
)
