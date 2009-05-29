import FWCore.ParameterSet.Config as cms

#Cono 0.5 Seed 1.0
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone5GenJetsSeed10 = cms.EDProducer("IterativeConeJetProducer",
    GenJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5GenJet'),
    coneRadius = cms.double(0.5)
)

iterativeCone5ChgGenJetsSeed10 = cms.EDProducer("IterativeConeJetProducer",
    GenJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5ChgGenJet'),
    coneRadius = cms.double(0.5)
)

iterativeCone5BasicJetsSeed10 = cms.EDProducer("IterativeConeJetProducer",
    IconeJetParameters,
    src = cms.InputTag("goodTracks"),
    coneRadius = cms.double(0.5),
    inputEtMin = cms.double(0.0),
    alias = cms.untracked.string('IC5BasicJet'),
    inputEMin = cms.double(0.0),
    jetType = cms.untracked.string('BasicJet'),
    towerThreshold = cms.double(1.0)
)

UEAnalysisJetsOnlyMC = cms.Sequence(iterativeCone5GenJetsSeed10*iterativeCone5ChgGenJetsSeed10)
UEAnalysisJetsOnlyReco = cms.Sequence(iterativeCone5BasicJetsSeed10)
UEAnalysisJets = cms.Sequence(UEAnalysisJetsOnlyMC*UEAnalysisJetsOnlyReco)
iterativeCone5GenJetsSeed10.seedThreshold = 1.0
iterativeCone5GenJetsSeed10.src = 'goodParticles'
iterativeCone5ChgGenJetsSeed10.seedThreshold = 1.0
iterativeCone5ChgGenJetsSeed10.src = 'chargeParticles'

