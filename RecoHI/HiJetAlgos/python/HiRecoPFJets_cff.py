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


hiRecoPFJets = cms.Sequence(
    PFTowers
    *akPu3PFJets*akPu4PFJets*akPu5PFJets
    )

