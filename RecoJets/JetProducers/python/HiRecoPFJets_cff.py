import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import AnomalousCellParameters
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import HiPFJetParameters,  MultipleAlgoIteratorBlock

#pseudo towers for noise suppression background subtraction
PFTowers = cms.EDProducer("ParticleTowerProducer",
                          src = cms.InputTag("particleFlow"),
                          useHF = cms.bool(False)
                          )

akPu5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
akPu5PFJets.jetType = cms.string('BasicJet')
akPu5PFJets.doPVCorrection = False
akPu5PFJets.doPUOffsetCorr = True
akPu5PFJets.subtractorName = cms.string("MultipleAlgoIterator")
akPu5PFJets.src = cms.InputTag('PFTowers')
akPu5PFJets.doAreaFastjet = False

akPu5PFJets.puPtMin = cms.double(25)
akPu3PFJets = akPu5PFJets.clone(rParam       = cms.double(0.3), puPtMin = 15)
akPu4PFJets = akPu5PFJets.clone(rParam       = cms.double(0.4), puPtMin = 20)

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets

kt4PFJetsForRhoHI = kt4PFJets.clone()    #note that this is also defined with different name for pA in RecoJets/Configuration/python/RecoJetsGlobal_cff.py. What is the right place to this stuff?
kt4PFJetsForRhoHI.src = cms.InputTag('particleFlow')
kt4PFJetsForRhoHI.doAreaFastjet = cms.bool(True)
kt4PFJetsForRhoHI.jetPtMin      = cms.double(0.0)
kt4PFJetsForRhoHI.GhostArea     = cms.double(0.005)

from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
hiFJRhoProducer.jetSource = cms.InputTag('kt4PFJetsForRhoHI')
hiFJRhoProducer.etaRanges = cms.vdouble(-5., -3., -2.1, -1.3, 1.3, 2.1, 3., 5.)

akCs4PFJets = cms.EDProducer(
    "CSJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm  = cms.string("AntiKt"),
    rParam        = cms.double(0.4),
    etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
    rho       = cms.InputTag('hiFJRhoProducer','mapToRho'),
    rhom      = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
    csRParam  = cms.double(-1.),
    csAlpha   = cms.double(2.),
    writeJetsWithConst = cms.bool(True),
    jetCollInstanceName = cms.string("pfParticlesCs")
)
akCs4PFJets.src           = cms.InputTag('particleFlow')
akCs4PFJets.doAreaFastjet = cms.bool(True)
akCs4PFJets.jetPtMin      = cms.double(0.0)
akCs4PFJets.useExplicitGhosts = cms.bool(True)
akCs4PFJets.GhostArea     = cms.double(0.005)

akCs3PFJets = akCs4PFJets.clone(rParam       = cms.double(0.3))

recoPFJetsHI = cms.Sequence(
    PFTowers
    *akPu3PFJets*akPu4PFJets*akPu5PFJets
    *kt4PFJetsForRhoHI
    *hiFJRhoProducer
    *akCs3PFJets*akCs4PFJets
    )

