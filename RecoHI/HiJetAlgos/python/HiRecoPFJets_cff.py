import FWCore.ParameterSet.Config as cms

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiPFJetParameters_cff import *


ak4PFJets = cms.EDProducer(
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
    rParam       = cms.double(0.4),
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

ic4PFJets = ic5PFJets.clone()
ic4PFJets.rParam       = cms.double(0.4)

ak3PFJets = ak4PFJets.clone()
ak3PFJets.rParam       = cms.double(0.3)

ak4PFJets = ak4PFJets.clone()
ak4PFJets.rParam       = cms.double(0.4)

ak7PFJets = ak4PFJets.clone()
ak7PFJets.rParam       = cms.double(0.7)

kt3PFJets = kt4PFJets.clone()
kt3PFJets.rParam       = cms.double(0.3)

kt6PFJets = kt4PFJets.clone()
kt6PFJets.rParam       = cms.double(0.6)


#hiRecoPFJets = cms.Sequence(ic5PFJets)
#hiRecoAllPFJets = cms.Sequence(ic5PFJets + ak4PFJets + ak7PFJets + kt4PFJets + kt6PFJets)


