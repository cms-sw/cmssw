import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *

from RecoJets.JetProducers.CaloJetPileupSubtractionParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
#from JetMETCorrections.Configuration.MCJetCorrections152_cff import *

CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

caloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    e = cms.double(0.0),
    verbose = cms.untracked.int32(0),
    pt = cms.double(0.0),
    minimumE = cms.double(0.0),
    minimumEt = cms.double(0.0),
    et = cms.double(0.0)
)

iterativeConePu5CaloJets = cms.EDProducer("IterativeConePilupSubtractionJetProducer",
    CaloJetPileupSubtractionParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJetPileupSubtraction'),
    coneRadius = cms.double(0.5)
)

# REPLACE with UP-TO-DATE Corrections
#MCJetCorJetIconePu5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeConePu5CaloJets"),
#    correctors = cms.vstring('MCJetCorrectorIcone5'),
#    alias = cms.untracked.string('MCJetCorJetIconePu5')
#)

iterativeCone5HiGenJets = cms.EDProducer("IterativeConeHiGenJetProducer",
                                         IconeJetParameters,
                                         inputEtMin = cms.double(0.0),                                        
                                         inputEMin = cms.double(0.0),                                        
                                         src = cms.InputTag("hiGenParticles"),
                                         jetType = cms.string('GenJet'),                                        
                                         alias = cms.untracked.string('IC5HiGenJet'),
                                         coneRadius = cms.double(0.5)
                                         )

CaloJetPileupSubtractionParameters.inputEtJetMin = 10.

runjets = cms.Sequence(caloTowersRec*caloTowers*iterativeConePu5CaloJets)
