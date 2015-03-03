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
                                       tableLabel = cms.string("UETable_Calo"),
                                       doEqualize = cms.bool(True),
                                       equalizeThreshold0 = cms.double(5.0),
                                       equalizeThreshold1 = cms.double(35.0),
                                       equalizeR = cms.double(0.4),
				       useTextTable = cms.bool(False),
				       jetCorrectorFormat = cms.bool(True),
                                       isCalo = cms.bool(True),
                                       etaBins = cms.int32(15),
                                       fourierOrder = cms.int32(5)
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
    dropZeros = cms.bool(True),
    doAreaFastjet = False
    )

#
akVs1CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.1))
akVs2CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.2))
akVs3CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.3))
akVs4CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.4))
akVs6CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.6))
akVs7CaloJets = akVs5CaloJets.clone(rParam       = cms.double(0.7))

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
hiRecoJets = cms.Sequence(
    caloTowersRec*caloTowers*
    iterativeConePu5CaloJets*
    akPu3CaloJets*akPu4CaloJets*akPu5CaloJets*
    voronoiBackgroundCalo*
    akVs2CaloJets*akVs3CaloJets*akVs4CaloJets*akVs5CaloJets
    )

## Extended Sequence
hiRecoAllJets = cms.Sequence(
    caloTowersRec*caloTowers*iterativeConePu5CaloJets
    *ak1CaloJets*ak2CaloJets*ak3CaloJets*ak4CaloJets*ak5CaloJets*ak6CaloJets*ak7CaloJets
    *akPu1CaloJets*akPu2CaloJets*akPu3CaloJets*akPu4CaloJets*akPu5CaloJets*akPu6CaloJets*akPu7CaloJets*
    ktPu4CaloJets*ktPu6CaloJets
    *voronoiBackgroundCalo
    *akVs1CaloJets*akVs2CaloJets*akVs3CaloJets*akVs4CaloJets*akVs5CaloJets*akVs6CaloJets*akVs7CaloJets    
    )


