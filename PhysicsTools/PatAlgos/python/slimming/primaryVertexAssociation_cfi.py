import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.sortedPFPrimaryVertices_cfi import sortedPFPrimaryVertices

primaryVertexAssociation = sortedPFPrimaryVertices.clone(
  qualityForPrimary = cms.int32(2),  
  produceSortedVertices = cms.bool(False),
  producePileUpCollection  = cms.bool(False),  
  produceNoPileUpCollection = cms.bool(False)
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    primaryVertexAssociation,
    vertices=cms.InputTag("offlinePrimaryVertices4D"),
    trackTimeTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
    trackTimeResoTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
    assignment=dict(useTiming=True),
)
