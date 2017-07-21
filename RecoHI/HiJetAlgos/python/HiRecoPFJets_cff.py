import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *

#pseudo towers for noise suppression background subtraction
PFTowers = cms.EDProducer("ParticleTowerProducer",
                          src = cms.InputTag("particleFlowTmp"),
                          useHF = cms.bool(False)
                          )



ak5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
ak5PFJets.src = cms.InputTag('particleFlowTmp')

akPu5PFJets = ak5PFJets.clone(
    jetType = cms.string('BasicJet'),
    doPVCorrection = False,
    doPUOffsetCorr = True,
    subtractorName = cms.string("MultipleAlgoIterator"),    
    src = cms.InputTag('PFTowers'),
    doAreaFastjet = False
    )



akPu5PFJets.puPtMin = cms.double(25)
akPu1PFJets = akPu5PFJets.clone(rParam       = cms.double(0.1), puPtMin = 10)
akPu2PFJets = akPu5PFJets.clone(rParam       = cms.double(0.2), puPtMin = 10)
akPu3PFJets = akPu5PFJets.clone(rParam       = cms.double(0.3), puPtMin = 15)
akPu4PFJets = akPu5PFJets.clone(rParam       = cms.double(0.4), puPtMin = 20)
akPu6PFJets = akPu5PFJets.clone(rParam       = cms.double(0.6), puPtMin = 30)
akPu7PFJets = akPu5PFJets.clone(rParam       = cms.double(0.7), puPtMin = 35)

kt4PFJetsForRho = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4)
)
kt4PFJetsForRho.src = cms.InputTag('particleFlowTmp')
kt4PFJetsForRho.doAreaFastjet = cms.bool(True)
kt4PFJetsForRho.jetPtMin      = cms.double(0.0)
kt4PFJetsForRho.GhostArea     = cms.double(0.005)

hiFJRhoProducer = cms.EDProducer('HiFJRhoProducer',
                                 jetSource = cms.InputTag('kt4PFJetsForRho'),
                                 nExcl = cms.int32(2),
                                 etaMaxExcl = cms.double(2.),
                                 ptMinExcl = cms.double(20.),
                                 nExcl2 = cms.int32(1),
                                 etaMaxExcl2 = cms.double(3.),
                                 ptMinExcl2 = cms.double(20.),
                                 etaRanges = cms.vdouble(-5., -3., -2.1, -1.3, 1.3, 2.1, 3., 5.)
)

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
akCs4PFJets.src           = cms.InputTag('particleFlowTmp')
akCs4PFJets.doAreaFastjet = cms.bool(True)
akCs4PFJets.jetPtMin      = cms.double(0.0)
akCs4PFJets.useExplicitGhosts = cms.bool(True)
akCs4PFJets.GhostArea     = cms.double(0.005)

akCs3PFJets = akCs4PFJets.clone(rParam       = cms.double(0.3))

hiRecoPFJets = cms.Sequence(
    PFTowers
    *akPu3PFJets*akPu4PFJets*akPu5PFJets
    *kt4PFJetsForRho*hiFJRhoProducer
    *akCs3PFJets*akCs4PFJets
    )

