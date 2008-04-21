import FWCore.ParameterSet.Config as cms

import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau1 = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau2 = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau3 = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau4 = copy.deepcopy(caloTowerMakerHLT)
from RecoJets.JetProducers.IconeJetParameters_cfi import *
from RecoJets.JetProducers.CaloJetParameters_cfi import *
icone5Tau1 = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJet'),
    coneRadius = cms.double(0.5)
)

icone5Tau2 = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJet'),
    coneRadius = cms.double(0.5)
)

icone5Tau3 = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJet'),
    coneRadius = cms.double(0.5)
)

icone5Tau4 = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJet'),
    coneRadius = cms.double(0.5)
)

caloTausCreator = cms.Sequence(cms.SequencePlaceholder("doCalo")*caloTowersTau1*icone5Tau1*caloTowersTau2*icone5Tau2*caloTowersTau3*icone5Tau3*caloTowersTau4*icone5Tau4)
caloTowersTau1.TauId = 0
caloTowersTau1.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau1.towers = 'towerMakerForAll'
caloTowersTau2.TauId = 1
caloTowersTau2.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau2.towers = 'towerMakerForAll'
caloTowersTau3.TauId = 2
caloTowersTau3.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau3.towers = 'towerMakerForAll'
caloTowersTau4.TauId = 3
caloTowersTau4.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau4.towers = 'towerMakerForAll'
icone5Tau1.src = 'caloTowersTau1'
icone5Tau2.src = 'caloTowersTau2'
icone5Tau3.src = 'caloTowersTau3'
icone5Tau4.src = 'caloTowersTau4'

