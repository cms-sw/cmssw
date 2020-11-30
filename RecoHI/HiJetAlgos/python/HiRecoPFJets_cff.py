import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *

#pseudo towers for noise suppression background subtraction
import RecoHI.HiJetAlgos.particleTowerProducer_cfi as _mod
PFTowers = _mod.particleTowerProducer.clone(useHF = True)

#dummy sequence to speed-up reconstruction in pp_on_AA era
pfEmptyCollection = cms.EDFilter('GenericPFCandidateSelector',
                                 src = cms.InputTag('particleFlow'),
                                 cut = cms.string("pt<0")
                                )

ak5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
)
ak5PFJets.src = 'particleFlow'

akPu5PFJets = ak5PFJets.clone(
    jetType        = 'BasicJet',
    doPVCorrection = False,
    doPUOffsetCorr = True,
    subtractorName = "MultipleAlgoIterator",
    src            = 'PFTowers',
    doAreaFastjet  = False,
    puPtMin        = cms.double(25)
)

akPu1PFJets = akPu5PFJets.clone(rParam = 0.1, puPtMin = 10)
akPu2PFJets = akPu5PFJets.clone(rParam = 0.2, puPtMin = 10)
akPu3PFJets = akPu5PFJets.clone(rParam = 0.3, puPtMin = 15)
akPu4PFJets = akPu5PFJets.clone(rParam = 0.4, puPtMin = 20)
akPu6PFJets = akPu5PFJets.clone(rParam = 0.6, puPtMin = 30)
akPu7PFJets = akPu5PFJets.clone(rParam = 0.7, puPtMin = 35)

hiPFCandCleanerforJets = cms.EDFilter('GenericPFCandidateSelector',
                                src = cms.InputTag('particleFlow'),
                                cut = cms.string("pt>5 && abs(eta)< 2")
                                )

ak4PFJetsForFlow = akPu5PFJets.clone(
   Ghost_EtaMax = 5.0,
   Rho_EtaMax   = 4.4,
   doRhoFastjet = False,
   jetPtMin     = 15.0,
   nSigmaPU     = 1.0,
   rParam       = 0.4,
   radiusPU     = 0.5,
   src          = "hiPFCandCleanerforJets",
)

kt4PFJetsForRho = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4)
)
kt4PFJetsForRho.src           = 'particleFlow'
kt4PFJetsForRho.doAreaFastjet = True
kt4PFJetsForRho.jetPtMin      = 0.0
kt4PFJetsForRho.GhostArea     = 0.005

from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer

import RecoHI.HiJetAlgos.hiFJRhoFlowModulationProducer_cfi as _mod
hiFJRhoFlowModulation = _mod.hiFJRhoFlowModulationProducer.clone()

import RecoHI.HiJetAlgos.hiPuRhoProducer_cfi as _mod
hiPuRho = _mod.hiPuRhoProducer.clone()

akCs4PFJets = cms.EDProducer(
    "CSJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm  = cms.string("AntiKt"),
    rParam        = cms.double(0.4),
    etaMap = cms.InputTag('hiPuRho', 'mapEtaEdges'),
    rho = cms.InputTag('hiPuRho', 'mapToRho'),
    rhom = cms.InputTag('hiPuRho', 'mapToRhoM'),
    csRParam  = cms.double(-1.),
    csAlpha   = cms.double(2.),
    writeJetsWithConst = cms.bool(True),
    useModulatedRho = cms.bool(False),
    rhoFlowFitParams = cms.InputTag('hiFJRhoFlowModulation', 'rhoFlowFitParams'),
    jetCollInstanceName = cms.string("pfParticlesCs"),
)
akCs4PFJets.src               = 'particleFlow'
akCs4PFJets.doAreaFastjet     = True
akCs4PFJets.jetPtMin          = 0.0
akCs4PFJets.useExplicitGhosts = cms.bool(True)
akCs4PFJets.GhostArea         = 0.005

akCs3PFJets = akCs4PFJets.clone(rParam = 0.3)

hiRecoPFJetsTask = cms.Task(
                           PFTowers,
                           akPu3PFJets,
                           akPu4PFJets,
                           akPu5PFJets,
                           hiPFCandCleanerforJets,
                           kt4PFJetsForRho,
                           ak4PFJetsForFlow,
                           hiFJRhoProducer,
                           hiPuRho,
                           hiFJRhoFlowModulation,
                           akCs3PFJets,
                           akCs4PFJets
    )
hiRecoPFJets = cms.Sequence(hiRecoPFJetsTask)

from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toModify(akCs4PFJets,src = 'cleanedParticleFlow')
run2_miniAOD_pp_on_AA_103X.toModify(PFTowers,src = 'cleanedParticleFlow')
