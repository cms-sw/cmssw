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

## background for HF/Voronoi-style subtraction
voronoiBackgroundCalo = cms.EDProducer('VoronoiBackgroundProducer',
                                       src = cms.InputTag('towerMaker'),
                                       equalizeR = cms.double(0.3)
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


## HF/Vornoi background subtracton algos

akVs5CaloJets = akPu5CaloJets.clone(
    subtractorName = cms.string("VoronoiSubtractor"),
    bkg = cms.InputTag("voronoiBackgroundCalo"),
    dropZeros = cms.untracked.bool(True),
    doAreaFastjet = False
    )

akVs2CaloJet = akVs5CaloJets.clone(rParam       = cms.double(0.2))
akVs3CaloJet = akVs5CaloJets.clone(rParam       = cms.double(0.3))
akVs4CaloJet = akVs5CaloJets.clone(rParam       = cms.double(0.4))
akVs6CaloJet = akVs5CaloJets.clone(rParam       = cms.double(0.6))
akVs7CaloJet = akVs5CaloJets.clone(rParam       = cms.double(0.7))


## Default Sequence
hiRecoJets = cms.Sequence(
    caloTowersRec*caloTowers*iterativeConePu5CaloJets
    *voronoiBackgroundCalo*akVs5CaloJets
    *akVs2CaloJet*akVs2CaloJet*akVs4CaloJet*akVs6CaloJet*akVs7CaloJet
    )

## Extended Sequence
hiRecoAllJets = cms.Sequence(
    caloTowersRec*caloTowers*iterativeConePu5CaloJets*akPu5CaloJets*akPu7CaloJets*ktPu4CaloJets*ktPu6CaloJets
    *voronoiBackgroundCalo*akVs5CaloJets
    )


