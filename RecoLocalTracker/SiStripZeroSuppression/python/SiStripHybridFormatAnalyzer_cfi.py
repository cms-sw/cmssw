import FWCore.ParameterSet.Config as cms

SiStripHybridFormatAnalyzer = cms.EDAnalyzer("SiStripHybridFormatAnalyzer",

    srcDigis =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    srcAPVCM =  cms.InputTag('siStripZeroSuppression','APVCMVirginRaw'),
    nModuletoDisplay = cms.uint32(10000),
    plotAPVCM	= cms.bool(True)
   
)
 