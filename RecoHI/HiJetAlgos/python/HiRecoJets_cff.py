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
                                          subtractorName = cms.string("MultipleAlgoIterator"),
                                          doFastJetNonUniform = cms.bool(True),
                                          puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
                                          puWidth = cms.double(0.5)
                                          )

iterativeConePu5CaloJets.doPUOffsetCorr = True
iterativeConePu5CaloJets.doAreaFastjet = True
iterativeConePu5CaloJets.doRhoFastjet = False
iterativeConePu5CaloJets.doPVCorrection = False
iterativeConePu5CaloJets.jetPtMin = 10
iterativeConePu5CaloJets.radiusPU = 0.5

iterativeConePu7CaloJets = cms.EDProducer("FastjetJetProducer",
                                          CaloJetParameters,
                                          AnomalousCellParameters,
                                          jetAlgorithm = cms.string("IterativeCone"),
                                          rParam       = cms.double(0.7),
                                          subtractorName = cms.string("MultipleAlgoIterator"),
                                          doFastJetNonUniform = cms.bool(True),
                                          puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
                                          puWidth = cms.double(0.5)
                                          )

iterativeConePu7CaloJets.doPUOffsetCorr = True
iterativeConePu7CaloJets.doAreaFastjet = True
iterativeConePu7CaloJets.doRhoFastjet = False
iterativeConePu7CaloJets.doPVCorrection = False
iterativeConePu7CaloJets.jetPtMin = 10
iterativeConePu7CaloJets.radiusPU = 0.7

ktPu4CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4),
    subtractorName = cms.string("MultipleAlgoIterator"),
    doFastJetNonUniform = cms.bool(True),
    puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
    puWidth = cms.double(0.5)
    )

ktPu4CaloJets.doPUOffsetCorr = True
ktPu4CaloJets.doAreaFastjet = True
ktPu4CaloJets.doRhoFastjet = False
ktPu4CaloJets.doPVCorrection = False
ktPu4CaloJets.jetPtMin = 10
ktPu4CaloJets.radiusPU = 0.5

ktPu6CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.6),
    subtractorName = cms.string("MultipleAlgoIterator"),
    doFastJetNonUniform = cms.bool(True),
    puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
    puWidth = cms.double(0.5)
    )

ktPu6CaloJets.doPUOffsetCorr = True
ktPu6CaloJets.doAreaFastjet = True
ktPu6CaloJets.doRhoFastjet = False
ktPu6CaloJets.doPVCorrection = False
ktPu6CaloJets.jetPtMin = 10
ktPu6CaloJets.radiusPU = 0.7

akPu5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5),
    subtractorName = cms.string("MultipleAlgoIterator"),
    doFastJetNonUniform = cms.bool(True),
    puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
    puWidth = cms.double(0.5)
    )

akPu5CaloJets.doPUOffsetCorr = True
akPu5CaloJets.doAreaFastjet = True
akPu5CaloJets.doRhoFastjet = False
akPu5CaloJets.doPVCorrection = False
akPu5CaloJets.jetPtMin = 10
akPu5CaloJets.radiusPU = 0.5

akPu7CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.7),
    subtractorName = cms.string("MultipleAlgoIterator"),
    doFastJetNonUniform = cms.bool(True),
    puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
    puWidth = cms.double(0.5)
    )

akPu7CaloJets.doPUOffsetCorr = True
akPu7CaloJets.doAreaFastjet = True
akPu7CaloJets.doRhoFastjet = False
akPu7CaloJets.doPVCorrection = False
akPu7CaloJets.jetPtMin = 10
akPu7CaloJets.radiusPU = 0.7

ic5CaloJets = akPu5CaloJets.clone()
ic5CaloJets.doRhoFastjet = True
ic5CaloJets.doPUOffsetCorr = False

ic7CaloJets = akPu7CaloJets.clone()
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

hiRecoJets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets)
hiRecoAllJets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets+iterativeConePu7CaloJets+ic5CaloJets+ic7CaloJets+akPu5CaloJets+akPu7CaloJets+ak5CaloJets+ak7CaloJets + ktPu4CaloJets + ktPu6CaloJets +  kt4CaloJets + kt6CaloJets)


