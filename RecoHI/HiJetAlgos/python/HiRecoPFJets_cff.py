import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *

#pseudo towers for noise suppression background subtraction
PFTowers = cms.EDProducer("ParticleTowerProducer",
                          src = cms.InputTag("particleFlow"),
                          useHF = cms.bool(False)
                          )

#dummy sequence to speed-up reconstruction in pp_on_AA era
pfNoPileUpJMEHI = cms.EDFilter('GenericPFCandidateSelector',
                                src = cms.InputTag('particleFlow'),
                                cut = cms.string("pt>9999")
                                )

ak5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
ak5PFJets.src = cms.InputTag('particleFlow')

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

hiPFCandCleanerforJets = cms.EDFilter('GenericPFCandidateSelector',
                                src = cms.InputTag('particleFlow'),
                                cut = cms.string("pt>5 && abs(eta)< 2")
                                )

ak4PFJetsForFlow = akPu5PFJets.clone(
   Ghost_EtaMax = 5.0,
   Rho_EtaMax = 4.4,
   doRhoFastjet = False,
   jetPtMin = 15.0,
   nSigmaPU = cms.double(1.0),
   rParam = 0.4,
   radiusPU = cms.double(0.5),
   src = "hiPFCandCleanerforJets",
)

# ak4PFJetsForFlow = akPu5PFJets.clone(
#     Ghost_EtaMax = cms.double(5.0),
#     Rho_EtaMax = cms.double(4.4),
#     doRhoFastjet = cms.bool(False),
#     jetPtMin = cms.double(15.0),
#     nSigmaPU = cms.double(1.0),
#     rParam = cms.double(0.4),
#     radiusPU = cms.double(0.5),
#     src = cms.InputTag("hiPFCandCleanerforJets"),
# )

kt4PFJetsForRho = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4)
)

kt4PFJetsForRho.src = cms.InputTag('particleFlow')
kt4PFJetsForRho.doAreaFastjet = cms.bool(True)
kt4PFJetsForRho.jetPtMin      = cms.double(0.0)
kt4PFJetsForRho.GhostArea     = cms.double(0.005)

from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
#from RecoHI.HiJetAlgos.hiPuRhoProducer_cfi import hiPuRhoProducer
#from RecoHI.HiJetAlgos.hiFJRhoFlowModulationProducer_cfi import hiFJRhoFlowModulationProducer
#from RecoHI.HiJetAlgos.hiPuRhoProducer import hiPuRhoProducer
#from RecoHI.HiJetAlgos.hiFJRhoFlowModulationProducer import hiFJRhoFlowModulationProducer
#from RecoHI.HiJetAlgos.hiPFCandCleaner_cfi import hiPFCandCleaner

hiFJRhoFlowModulationProducer = cms.EDProducer(
    'HiFJRhoFlowModulationProducer',
    pfCandSource = cms.InputTag('particleFlow'),
    doJettyExclusion = cms.bool(False),
    doFreePlaneFit = cms.bool(False),
    doEvtPlane = cms.bool(False),
    doFlatTest = cms.bool(False),
    jetTag = cms.InputTag("ak4PFJets"),
    EvtPlane = cms.InputTag("hiEvtPlane"),
    evtPlaneLevel = cms.int32(0)
    )	

hiPuRhoProducer = cms.EDProducer(
    'HiPuRhoProducer',
    dropZeroTowers = cms.bool(True),
    medianWindowWidth = cms.int32(2),
    minimumTowersFraction = cms.double(0.7),
    nSigmaPU = cms.double(1.0),
    puPtMin = cms.double(15.0),
    rParam = cms.double(.3),
    radiusPU = cms.double(.5),
    src = cms.InputTag('PFTowers'),
    towSigmaCut = cms.double(5.), 
    )

akCs4PFJets = cms.EDProducer(
    "CSJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm  = cms.string("AntiKt"),
    rParam        = cms.double(0.4),
    etaMap = cms.InputTag('hiPuRhoProducer', 'mapEtaEdges'),
    rho = cms.InputTag('hiPuRhoProducer', 'mapToRho'),
    rhom = cms.InputTag('hiPuRhoProducer', 'mapToRhoM'),
    csRParam  = cms.double(-1.),
    csAlpha   = cms.double(2.),
    writeJetsWithConst = cms.bool(True),
    useModulatedRho = cms.bool(True),
    rhoFlowFitParams = cms.InputTag('hiFJRhoFlowModulationProducer', 'rhoFlowFitParams'),
    jetCollInstanceName = cms.string("pfParticlesCs"),
)
akCs4PFJets.src           = cms.InputTag('particleFlow')
akCs4PFJets.doAreaFastjet = cms.bool(True)
akCs4PFJets.jetPtMin      = cms.double(0.0)
akCs4PFJets.useExplicitGhosts = cms.bool(True)
akCs4PFJets.GhostArea     = cms.double(0.005)

akCs3PFJets = akCs4PFJets.clone(rParam       = cms.double(0.3))

hiRecoPFJetsTask = cms.Task(
                           PFTowers,
                           akPu3PFJets,
                           akPu4PFJets,
                           akPu5PFJets,
                           hiPFCandCleanerforJets,
                           kt4PFJetsForRho,
                           ak4PFJetsForFlow,
                           hiFJRhoProducer,
                           hiPuRhoProducer,
                           hiFJRhoFlowModulationProducer,
                           akCs3PFJets,
                           akCs4PFJets
    )
hiRecoPFJets = cms.Sequence(hiRecoPFJetsTask)



