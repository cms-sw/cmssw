import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

SiStripBaselineAnalyzer = cms.EDAnalyzer("SiStripBaselineAnalyzer",

    Algorithms = DefaultAlgorithms,
    srcBaseline =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINE'),
    srcAPVCM  =  cms.InputTag('siStripZeroSuppression','APVCM'),
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    nModuletoDisplay = cms.uint32(1000),
    plotClusters = cms.bool(False),
    plotBaseline = cms.bool(True),
    plotRawDigi	= cms.bool(True),
    plotAPVCM	= cms.bool(True),
    plotPedestals = cms.bool(True)
)
