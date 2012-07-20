import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *


ak5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5),
    )

kt4PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.5),
    )

ic5PFJets = cms.EDProducer(
    "FastjetJetProducer",
    HiPFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("IterativeCone"),
    rParam       = cms.double(0.5),
    subtractorName = cms.string("MultipleAlgoIterator"),
    )



ic3PFJets = ic5PFJets.clone()
ic3PFJets.rParam       = cms.double(0.3)
ic3PFJets.radiusPU = 0.3

ic4PFJets = ic5PFJets.clone()
ic4PFJets.rParam       = cms.double(0.4)
ic4PFJets.radiusPU = 0.4

ak3PFJets = ak5PFJets.clone()
ak3PFJets.rParam       = cms.double(0.3)
ak3PFJets.radiusPU = 0.3

ak4PFJets = ak5PFJets.clone()
ak4PFJets.rParam       = cms.double(0.4)
ak4PFJets.radiusPU = 0.4

ak7PFJets = ak5PFJets.clone()
ak7PFJets.rParam       = cms.double(0.7)
ak7PFJets.radiusPU = 0.7

kt3PFJets = kt4PFJets.clone()
kt3PFJets.rParam       = cms.double(0.3)
kt3PFJets.radiusPU = 0.3

kt6PFJets = kt4PFJets.clone()
kt6PFJets.rParam       = cms.double(0.6)
kt6PFJets.radiusPU = 0.6



hiRecoPFJets = cms.Sequence(ic5PFJets)
hiRecoAllPFJets = cms.Sequence(ic5PFJets + ak5PFJets + ak7PFJets + kt4PFJets + kt6PFJets)


