import FWCore.ParameterSet.Config as cms

siStripZeroSuppression = cms.EDFilter("SiStripZeroSuppression",
    RawDigiProducersList = cms.VPSet(cms.PSet(
        RawDigiProducer = cms.string('SiStripDigis'),
        RawDigiLabel = cms.string('VirginRaw')
    ), cms.PSet(
        RawDigiProducer = cms.string('SiStripDigis'),
        RawDigiLabel = cms.string('ProcessedRaw')
    ), cms.PSet(
        RawDigiProducer = cms.string('SiStripDigis'),
        RawDigiLabel = cms.string('ScopeMode')
    )),
    FEDalgorithm = cms.uint32(4),
    ZeroSuppressionMode = cms.string('SiStripFedZeroSuppression'),
    CutToAvoidSignal = cms.double(3.0), ##

    CommonModeNoiseSubtractionMode = cms.string('Median') ##Supported modes: Median, TT6, FastLinear

)


