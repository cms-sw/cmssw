import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.sortedPFPrimaryVertices_cfi import sortedPFPrimaryVertices
sortedPackedPrimaryVertices = cms.EDProducer("PackedCandidatePrimaryVertexSorter",
    sorting = cms.PSet(),
    assignment = sortedPFPrimaryVertices.assignment,
    particles = cms.InputTag("packedPFCandidates"),
    vertices= cms.InputTag("offlineSlimmedPrimaryVertices"),
    jets= cms.InputTag("slimmedJets"),
    qualityForPrimary = cms.int32(3),
    usePVMET = cms.bool(True),
    produceAssociationToOriginalVertices = cms.bool(True),
    produceSortedVertices = cms.bool(True),
    producePileUpCollection  = cms.bool(True),
    produceNoPileUpCollection = cms.bool(True),
)
