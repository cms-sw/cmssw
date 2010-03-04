import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *

from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

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

iterativeConePu5CaloJets = cms.EDProducer("FastjetJetProducer",
                                          CaloJetParameters,
                                          AnomalousCellParameters,
                                          jetAlgorithm = cms.string("IterativeCone"),
                                          rParam       = cms.double(0.5),
                                          )

iterativeConePu5CaloJets.doPUOffsetCorr = True
iterativeConePu5CaloJets.doPVCorrection = False
iterativeConePu5CaloJets.jetPtMin = 10


iterativeConePu7CaloJets = cms.EDProducer("FastjetJetProducer",
                                          CaloJetParameters,
                                          AnomalousCellParameters,
                                          jetAlgorithm = cms.string("IterativeCone"),
                                          rParam       = cms.double(0.7),
                                          )

iterativeConePu7CaloJets.doPUOffsetCorr = True
iterativeConePu7CaloJets.doPVCorrection = False
iterativeConePu7CaloJets.jetPtMin = 10

akPu5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )

akPu5CaloJets.doPUOffsetCorr = True
akPu5CaloJets.doPVCorrection = False
akPu5CaloJets.jetPtMin = 10

akPu7CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.7)
    )

akPu7CaloJets.doPUOffsetCorr = True
akPu7CaloJets.doPVCorrection = False
akPu7CaloJets.jetPtMin = 10

akFastPu5CaloJets = akPu5CaloJets.clone()
akFastPu5CaloJets.doPUOffsetCorr = False
akFastPu5CaloJets.doAreaFastjet = True
akFastPu5CaloJets.doRhoFastjet = True

akFastPu7CaloJets = akPu7CaloJets.clone()
akFastPu7CaloJets.doPUOffsetCorr = False
akFastPu7CaloJets.doAreaFastjet = True
akFastPu7CaloJets.doRhoFastjet = True


# REPLACE with UP-TO-DATE Corrections
#MCJetCorJetIconePu5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeConePu5CaloJets"),
#    correctors = cms.vstring('MCJetCorrectorIcone5'),
#    alias = cms.untracked.string('MCJetCorJetIconePu5')
#)

#iterativeCone5HiGenJets = cms.EDProducer("IterativeConeHiGenJetProducer",
#                                         IconeJetParameters,
#                                         inputEtMin = cms.double(0.0),                                        
#                                         inputEMin = cms.double(0.0),                                        
#                                         src = cms.InputTag("hiGenParticles"),
#                                         jetType = cms.string('GenJet'),                                        
#                                         alias = cms.untracked.string('IC5HiGenJet'),
#                                         coneRadius = cms.double(0.5)
#                                         )

hiRecoJets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets+iterativeConePu7CaloJets+akPu5CaloJets+akPu7CaloJets+akFastPu5CaloJets+akFastPu7CaloJets)
