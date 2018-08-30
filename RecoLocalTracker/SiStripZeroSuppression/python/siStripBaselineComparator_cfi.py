import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripBaselineComparator = cms.EDAnalyzer("SiStripBaselineComparator",

    srcClusters =  cms.InputTag('siStripClusters','siStripClusters'),
    srcClusters2 =  cms.InputTag('moddedsiStripClusters','siStripClusters'),
)
