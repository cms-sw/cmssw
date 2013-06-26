import FWCore.ParameterSet.Config as cms

siPixelCalibDigis = cms.EDProducer("SiPixelCalibDigiProducer",
    src = cms.InputTag("siPixelDigis"),
    # force the producer to use the event numbers as in the data. If false the producer will synchronize by using its own internal counter.
    useRealEventNumber = cms.bool(False),
    # setting ignoreNonPattern to true makes the production of CalibDigis significantly slower!
    ignoreNonPattern = cms.bool(True),
    # error output: it is possible to write out SiPixelRawDataErrors. Use this flag to turn this on. It will not save pixel info, only det id and fed ID.
    includeErrors = cms.untracked.bool(True),
    # the error type numbers are defined in DataFormats/SiPixelRawData/src/SiPixelRawDataError.cc (SetError() method). We will use 31 as default, this one is defined as 'event number mismatch..'
    errorTypeNumber = cms.untracked.int32(31),
    label = cms.string('siPixelCalibDigis'),
    instance = cms.string('test'),
    # replace set checkPatternEachEvent to false to only check on pixel patterns when filling the calib digis. setting it to true means every event will be checked. does nothing when ignoreNonPattern is false
    checkPatternEachEvent = cms.bool(False)
)


