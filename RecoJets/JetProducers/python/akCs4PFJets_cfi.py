import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

akCs4PFJets = cms.EDProducer(
    "CSJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm  = cms.string("AntiKt"),
    rParam        = cms.double(0.4),
    etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
    rho       = cms.InputTag('hiFJRhoProducer','mapToRho'),
    rhom      = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
    csAlpha   = cms.double(1.),
    writeJetsWithConst = cms.bool(True),
    jetCollInstanceName = cms.string("pfParticlesCs")
)
akCs4PFJets.src           = cms.InputTag('particleFlowTmp')
akCs4PFJets.doAreaFastjet = cms.bool(True)
akCs4PFJets.jetPtMin      = cms.double(0.0)
akCs4PFJets.useExplicitGhosts = cms.bool(True)
