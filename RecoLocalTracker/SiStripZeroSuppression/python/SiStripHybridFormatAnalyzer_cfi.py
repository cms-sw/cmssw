import FWCore.ParameterSet.Config as cms

SiStripHybridFormatAnalyzer = cms.EDAnalyzer("SiStripHybridFormatAnalyzer",

    Algorithms = DefaultAlgorithms,
    srcDigis =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    nModuletoDisplay = cms.uint32(10000),
   
)
 