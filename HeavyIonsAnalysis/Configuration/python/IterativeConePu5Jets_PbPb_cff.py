import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from RecoJets.JetProducers.CaloJetPileupSubtractionParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
from JetMETCorrections.Configuration.MCJetCorrections152_cff import *
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

MCJetCorJetIconePu5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('MCJetCorJetIconePu5')
)

runjets = cms.Sequence(towerMaker*caloTowers*iterativeConePu5CaloJets*genJetParticles*iterativeCone5GenJets*MCJetCorJetIconePu5)
CaloJetPileupSubtractionParameters.inputEtJetMin = 10.

