import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *

#pseudo towers for noise suppression background subtraction
particlePseudoTowers = cms.EDProducer("ParticleTowerProducer",
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
    src = cms.InputTag('particlePseudoTowers'),
    doAreaFastjet = False
    )


akVs5PFJets = ak5PFJets.clone(
    doPVCorrection = False,
    doPUOffsetCorr = True,
    subtractorName = cms.string("VoronoiSubtractor"),
    bkg = cms.InputTag("voronoiBackgroundPF"),
    src = cms.InputTag('particleFlowTmp'),
    dropZeros = cms.untracked.bool(True),
    doAreaFastjet = False
    )

akVs2PFJet = akVs5PFJets.clone(rParam       = cms.double(0.2))
akVs3PFJet = akVs5PFJets.clone(rParam       = cms.double(0.3))
akVs4PFJet = akVs5PFJets.clone(rParam       = cms.double(0.4))
akVs6PFJet = akVs5PFJets.clone(rParam       = cms.double(0.6))
akVs7PFJet = akVs5PFJets.clone(rParam       = cms.double(0.7))


hiRecoPFJets = cms.Sequence(
    particlePseudoTowers*akPu5PFJets
    *voronoiBackgroundPF*akVs5PFJets
    *akVs2PFJet*akVs2PFJet*akVs4PFJet*akVs6PFJet*akVs7PFJet
    )
#hiRecoAllPFJets = cms.Sequence(ic5PFJets + ak5PFJets + ak7PFJets + kt4PFJets + kt6PFJets)


