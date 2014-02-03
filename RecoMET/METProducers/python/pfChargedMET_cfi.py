import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
particleFlowForChargedMET = cms.EDProducer(
    "ParticleFlowForChargedMETProducer",
    PFCollectionLabel = cms.InputTag("particleFlow"),
    PVCollectionLabel = cms.InputTag("offlinePrimaryVertices"),
    dzCut = cms.double(0.2),
    neutralEtThreshold = cms.double(-1.0)
    )

##____________________________________________________________________________||
pfChargedMET = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("particleFlowForChargedMET"),
    METType = cms.string('PFMET'),
    alias = cms.string('PFMET'),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('PFCandidateCollection'),
    calculateSignificance = cms.bool(False),
    )

##____________________________________________________________________________||
