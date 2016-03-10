import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.Configuration.collisionEventSelection_cff import *
from HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff import *

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import HBHENoiseFilterResultProducer
fHBHENoiseFilterResult = cms.EDFilter(
    'BooleanFlagFilter',
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResult'),
    reverseDecision = cms.bool(False)
)
fHBHENoiseFilterResultRun1 = fHBHENoiseFilterResult.clone(
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResultRun1'))
fHBHENoiseFilterResultRun2Loose = fHBHENoiseFilterResult.clone(
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResultRun2Loose'))
fHBHENoiseFilterResultRun2Tight = fHBHENoiseFilterResult.clone(
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResultRun2Tight'))
fHBHEIsoNoiseFilterResult = fHBHENoiseFilterResult.clone(
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHEIsoNoiseFilterResult'))
