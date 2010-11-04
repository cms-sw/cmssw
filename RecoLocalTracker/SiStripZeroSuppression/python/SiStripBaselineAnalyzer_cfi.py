import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

SiStripBaselineAnalyzer = cms.EDAnalyzer("SiStripBaselineAnalyzer",

    Algorithms = DefaultAlgorithms,
    outputFile = cms.untracked.string("HistoRoot.root"),
    srcBaseline =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINE'),
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    nModuletoDisplay = cms.uint32(150)
)
