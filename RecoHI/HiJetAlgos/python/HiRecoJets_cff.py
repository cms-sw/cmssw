import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiCaloJetParameters_cff import *

caloTowers = cms.EDProducer("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    e = cms.double(0.0),
    verbose = cms.untracked.int32(0),
    pt = cms.double(0.0),
    minimumE = cms.double(0.0),
    minimumEt = cms.double(0.0),
    et = cms.double(0.0)
)

## Iterative Cone
iterativeConePu5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("IterativeCone"),
    rParam       = cms.double(0.5)
    )
iterativeConePu5CaloJets.radiusPU = 0.5

iterativeConePu7CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("IterativeCone"),
    rParam       = cms.double(0.7)
    )
iterativeConePu7CaloJets.radiusPU = 0.7

## kT
ktPu4CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4)
    )
ktPu4CaloJets.radiusPU = 0.5

ktPu6CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.6)
    )
ktPu6CaloJets.radiusPU = 0.7

## anti-kT
akPu5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
akPu5CaloJets.radiusPU = 0.5

akPu7CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.7)
    )
akPu7CaloJets.radiusPU = 0.7

## Algos without offset pileup correction
ic5CaloJets = iterativeConePu5CaloJets.clone()
ic5CaloJets.doRhoFastjet = True
ic5CaloJets.doPUOffsetCorr = False

ic7CaloJets = iterativeConePu7CaloJets.clone()
ic7CaloJets.doRhoFastjet = True
ic7CaloJets.doPUOffsetCorr = False

ak5CaloJets = akPu5CaloJets.clone()
ak5CaloJets.doRhoFastjet = True
ak5CaloJets.doPUOffsetCorr = False

ak7CaloJets = akPu7CaloJets.clone()
ak7CaloJets.doRhoFastjet = True
ak7CaloJets.doPUOffsetCorr = False

kt4CaloJets = ktPu4CaloJets.clone()
kt4CaloJets.doRhoFastjet = True
kt4CaloJets.doPUOffsetCorr = False

kt6CaloJets = ktPu6CaloJets.clone()
kt6CaloJets.doRhoFastjet = True
kt6CaloJets.doPUOffsetCorr = False

## Default Sequence
hiRecoJets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets)

## Extended Sequence
hiRecoAllJets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets+iterativeConePu7CaloJets+ic5CaloJets+ic7CaloJets+akPu5CaloJets+akPu7CaloJets+ak5CaloJets+ak7CaloJets + ktPu4CaloJets + ktPu6CaloJets +  kt4CaloJets + kt6CaloJets)


