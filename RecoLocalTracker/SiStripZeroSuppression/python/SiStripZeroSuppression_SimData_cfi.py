import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
                                        
    Algorithms = DefaultAlgorithms,
    RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'), 
                                          cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                          cms.InputTag('simSiStripDigis','ScopeMode')),
                                        
    storeCM = cms.bool(False), 
    fixCM= cms.bool(False),                # put -999 into CM collection for "inspected" APV
                                        
    produceRawDigis = cms.bool(False),    # if mergeCollection is True, produceRawDigi is not considered
    produceCalculatedBaseline = cms.bool(False),
    produceBaselinePoints = cms.bool(False),
    storeInZScollBadAPV = cms.bool(True),
    produceHybridFormat = cms.bool(False)
)
