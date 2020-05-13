import FWCore.ParameterSet.Config as cms
from RecoMET.METProducers.pfMet_cfi import pfMet

##____________________________________________________________________________||
particleFlowForChargedMET = cms.EDProducer(
    "ParticleFlowForChargedMETProducer",
    PFCollectionLabel = cms.InputTag("particleFlow"),
    PVCollectionLabel = cms.InputTag("offlinePrimaryVertices"),
    dzCut = cms.double(0.2),
    neutralEtThreshold = cms.double(-1.0)
    )

##____________________________________________________________________________||
pfChMet = pfMet.clone(
    src = "particleFlowForChargedMET",
    alias = 'pfChMet',
    )

##____________________________________________________________________________||
