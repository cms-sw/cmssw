import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

SiStripBaselineAnalyzer = cms.EDAnalyzer("SiStripBaselineAnalyzer",

    Algorithms = DefaultAlgorithms,
    srcBaseline =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINEVirginRaw'),
    srcBaselinePoints =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINEPOINTSVirginRaw'),
    srcAPVCM  =  cms.InputTag('siStripZeroSuppression','APVCMVirginRaw'),
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    srcDigis =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    nModuletoDisplay = cms.uint32(10000),
    plotClusters = cms.bool(True),
    plotBaseline = cms.bool(True),
    plotBaselinePoints = cms.bool(False),
    plotRawDigi	= cms.bool(True),
    plotAPVCM	= cms.bool(True),
    plotPedestals = cms.bool(True),
    plotDigis = cms.bool(True)
)
