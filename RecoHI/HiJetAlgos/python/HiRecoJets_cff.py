import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *

## Default Parameter Sets
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoHI.HiJetAlgos.HiCaloJetParameters_cff import *

## Calo Towers
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

caloTowers = cms.EDProducer("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    e = cms.double(0.0),
    verbose = cms.untracked.int32(0),
    pt = cms.double(0.0),
    minimumE = cms.double(0.0),
    minimumEt = cms.double(0.0),
    et = cms.double(0.0)
)

## Noise reducing PU subtraction algos

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


akPu5CaloJets.puPtMin = cms.double(10)
akPu1CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.1), puPtMin = 4)
akPu2CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.2), puPtMin = 4)
akPu3CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.3), puPtMin = 6)
akPu4CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.4), puPtMin = 8)
akPu6CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.6), puPtMin = 12)
akPu7CaloJets = akPu5CaloJets.clone(rParam       = cms.double(0.7), puPtMin = 14)

ak5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
ak5CaloJets.doPUOffsetCorr = False
ak1CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.1))
ak2CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.2))
ak3CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.3))
ak4CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.4))
ak6CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.6))
ak7CaloJets = ak5CaloJets.clone(rParam       = cms.double(0.7))


## Default Sequence
hiRecoJetsTask = cms.Task(
    caloTowersRecTask,caloTowers,
    iterativeConePu5CaloJets,
    akPu3CaloJets,akPu4CaloJets,akPu5CaloJets
    )
hiRecoJets = cms.Sequence(hiRecoJetsTask)

## Extended Sequence
hiRecoAllJetsTask = cms.Task(
    caloTowersRecTask,caloTowers,iterativeConePu5CaloJets
    ,ak1CaloJets,ak2CaloJets,ak3CaloJets,ak4CaloJets,ak5CaloJets,ak6CaloJets,ak7CaloJets
    ,akPu1CaloJets,akPu2CaloJets,akPu3CaloJets,akPu4CaloJets,akPu5CaloJets,akPu6CaloJets,akPu7CaloJets,
    ktPu4CaloJets,ktPu6CaloJets
    )
hiRecoAllJets = cms.Sequence(hiRecoAllJetsTask)
