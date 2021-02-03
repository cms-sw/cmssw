import FWCore.ParameterSet.Config as cms

castorDigitizer = cms.PSet(
    accumulatorType = cms.string('CastorDigiProducer'),
    castor = cms.PSet(
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(4.1718),
        readoutFrameSize = cms.int32(6),
        samplingFactor = cms.double(0.062577),
        simHitToPhotoelectrons = cms.double(1000.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    ),
    doNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    hitsProducer = cms.InputTag("g4SimHits","CastorFI"),
    makeDigiSimLinks = cms.untracked.bool(False)
)