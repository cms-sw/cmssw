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

## anti-kT
akPu4CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    HiCaloJetParameters,
    AnomalousCellParameters,
    MultipleAlgoIteratorBlock,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4)
    )
akPu4CaloJets.radiusPU = 0.4
akPu4CaloJets.puPtMin = cms.double(8)

akPu3CaloJets = akPu4CaloJets.clone(rParam       = cms.double(0.3), puPtMin = 6)
akPu5CaloJets = akPu4CaloJets.clone(rParam       = cms.double(0.5), puPtMin = 10)

## Default Sequence
recoJetsHI = cms.Sequence(
    caloTowersRec*caloTowers*
    akPu3CaloJets*akPu4CaloJets*akPu5CaloJets
    )



