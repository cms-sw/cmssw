import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitSiStripHitConverterFromClustersDefault_cfi import mkFitSiStripHitConverterFromClustersDefault as _mkFitSiStripHitConverterFromClustersDefault

mkFitSiStripHitConverterFromClusters = _mkFitSiStripHitConverterFromClustersDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
