import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",

    Algorithms = DefaultAlgorithms,
    RawDigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','VirginRaw'), 
                                          cms.InputTag('siStripDigis','ProcessedRaw'),
                                          cms.InputTag('siStripDigis','ScopeMode')),

    storeCM = cms.bool(False), 
    fixCM= cms.bool(False),                # put -999 into CM collection for "inspected" APV

    produceRawDigis = cms.bool(False),     # if mergeCollection is True, produceRawDigi is not considered
    produceCalculatedBaseline = cms.bool(False),
    produceBaselinePoints = cms.bool(False),
    storeInZScollBadAPV = cms.bool(True),
    mergeCollections = cms.bool(False),
    doAPVRestore = cms.bool(False),
    useCMMeanMap= cms.bool(False)
)
