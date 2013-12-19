import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *

#pseudo towers for noise suppression background subtraction
PFTowers = cms.EDProducer("ParticleTowerProducer",
                          src = cms.InputTag("particleFlowTmp"),
                          useHF = cms.untracked.bool(False)
                          )

## background for HF/Voronoi-style subtraction
voronoiBackgroundPF = cms.EDProducer('VoronoiBackgroundProducer',
                                     src = cms.InputTag('particleFlowTmp'),
                                     equalizeR = cms.double(0.3)
                                     )



ak5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5),
    )


akPu5PFJets = ak5PFJets.clone(
    jetType = cms.string('BasicJet'),
    doPVCorrection = False,
    doPUOffsetCorr = True,
    subtractorName = cms.string("MultipleAlgoIterator"),    
    src = cms.InputTag('PFTowers'),
    doAreaFastjet = False
    )


akVs5PFJets = ak5PFJets.clone(
    doPVCorrection = False,
    doPUOffsetCorr = True,
    subtractorName = cms.string("VoronoiSubtractor"),
    bkg = cms.InputTag("voronoiBackgroundPF"),
    src = cms.InputTag('particleFlowTmp'),
    dropZeros = cms.untracked.bool(True),
    doAreaFastjet = False,
    puPtMin = cms.double(0)
    )

akVs2PFJets = akVs5PFJets.clone(rParam       = cms.double(0.2))
akVs3PFJets = akVs5PFJets.clone(rParam       = cms.double(0.3))
akVs4PFJets = akVs5PFJets.clone(rParam       = cms.double(0.4))
akVs6PFJets = akVs5PFJets.clone(rParam       = cms.double(0.6))
akVs7PFJets = akVs5PFJets.clone(rParam       = cms.double(0.7))

akPu5PFJets.puPtMin = cms.double(25)
akPu2PFJets = akPu5PFJets.clone(rParam       = cms.double(0.2), puPtMin = 10)
akPu3PFJets = akPu5PFJets.clone(rParam       = cms.double(0.3), puPtMin = 15)
akPu4PFJets = akPu5PFJets.clone(rParam       = cms.double(0.4), puPtMin = 20)
akPu6PFJets = akPu5PFJets.clone(rParam       = cms.double(0.6), puPtMin = 30)
akPu7PFJets = akPu5PFJets.clone(rParam       = cms.double(0.7), puPtMin = 35)


hiRecoPFJets = cms.Sequence(
    PFTowers
    *akPu3PFJets*akPu5PFJets
    *voronoiBackgroundPF*akVs5PFJets
    *akVs2PFJets*akVs3PFJets*akVs4PFJets*akVs6PFJets*akVs7PFJets
    )

hiRecoAllPFJets = cms.Sequence(hiRecoPFJets * akPu2PFJets*akPu4PFJets*akPu6PFJets*akPu7PFJets)


