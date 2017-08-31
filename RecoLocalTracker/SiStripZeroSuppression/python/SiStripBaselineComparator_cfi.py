import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

SiStripBaselineComparitor = cms.EDAnalyzer("SiStripBaselineComparitor",

    srcClusters =  cms.InputTag('siStripClusters','siStripClusters'),
    srcClusters2 =  cms.InputTag('moddedsiStripClusters','siStripClusters'),
)
