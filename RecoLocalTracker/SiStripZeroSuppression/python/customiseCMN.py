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

    process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
    
    process.siStripZeroSuppression.Algorithms.APVInspectMode = cms.string("NullFraction")
    process.siStripZeroSuppression.Algorithms.APVRestoreMode = cms.string("Flat")
    process.siStripZeroSuppression.Algorithms.restoreThreshold = cms.double(0.5)

    return process

##############################################################################
def customisePartialSuppress(process):

    process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
    process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
    process.siStripZeroSuppression.storeInZScollBadAPV = cms.bool(False)

    process.siStripZeroSuppression.Algorithms.APVInspectMode = cms.string("AbnormalBaseline")
    process.siStripZeroSuppression.Algorithms.APVRestoreMode = cms.string("PartialSuppress")
    process.siStripZeroSuppression.Algorithms.Fraction = cms.double(0.2)
    process.siStripZeroSuppression.Algorithms.Deviation = cms.uint32(25)


    return process

##############################################################################
def customiseTier0(process):

    process.siStripZeroSuppression.Algorithms.PedestalSubtractionFedMode = cms.bool(False)
    
    customiseIteratedMedian(process)
    
    process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
    process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
    process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(True)

    # these are the current defaults.
    process.siStripZeroSuppression.Algorithms.APVInspectMode = cms.string("BaselineFollower")
    process.siStripZeroSuppression.Algorithms.APVRestoreMode = cms.string("BaselineFollower")
    process.siStripZeroSuppression.Algorithms.DeltaCMThreshold = cms.uint32(20)
    process.siStripZeroSuppression.Algorithms.distortionThreshold = cms.uint32(40)
    process.siStripZeroSuppression.Algorithms.nSigmaNoiseDerTh = cms.uint32(4)
    process.siStripZeroSuppression.Algorithms.consecThreshold = cms.uint32(5)
    process.siStripZeroSuppression.Algorithms.hitStripThreshold = cms.uint32(40)    
    process.siStripZeroSuppression.Algorithms.nSmooth = cms.uint32(9)      
    process.siStripZeroSuppression.Algorithms.minStripsToFit = cms.uint32(4)     

    return process

##############################################################################
def customiseMergeCollections(process):

    process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag(
        cms.InputTag('siStripVRDigis','VirginRaw'),
        cms.InputTag('siStripVRDigis','ProcessedRaw'),
        cms.InputTag('siStripVRDigis','ScopeMode')
        )

    return process
