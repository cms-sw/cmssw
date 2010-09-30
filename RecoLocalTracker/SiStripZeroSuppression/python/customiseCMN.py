import FWCore.ParameterSet.Config as cms

##############################################################################
def customiseMedian(process):

    process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("Median")
    process.siStripZeroSuppression.storeCM = cms.bool(True)

    return process


##############################################################################
def customiseIteratedMedian(process):

    process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
    process.siStripZeroSuppression.Algorithms.CutToAvoidSignal = cms.double(2.0)
    process.siStripZeroSuppression.Algorithms.Iterations = cms.int32(3)
    process.siStripZeroSuppression.storeCM = cms.bool(True)

    return process

##############################################################################
def customisePercentile(process):

    process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("Percentile")
    process.siStripZeroSuppression.Algorithms.Percentile = cms.double(25.0)
    process.siStripZeroSuppression.storeCM = cms.bool(True)

    return process

##############################################################################
def customiseFlatAPVRestore(process):

    process.siStripZeroSuppression.Algorithms.APVRestoreMode = cms.string("Flat")
    process.siStripZeroSuppression.Algorithms.restoreThreshold = cms.double(0.5)

    return process

##############################################################################
def customisePartialSuppress(process):

    process.siStripZeroSuppression.Algorithms.APVRestoreMode = cms.string("PartialSuppress")
    process.siStripZeroSuppression.Algorithms.Fraction = cms.double(0.2)
    process.siStripZeroSuppression.Algorithms.Deviation = cms.int32(25)
    process.siStripZeroSuppression.produceRawDigis = cms.bool(True)

    return process

##############################################################################
def customiseMergeCollections(process):

    process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag(
        cms.InputTag('siStripVRDigis','VirginRaw'),
        cms.InputTag('siStripVRDigis','ProcessedRaw'),
        cms.InputTag('siStripVRDigis','ScopeMode')
        )

    return process
