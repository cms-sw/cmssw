import FWCore.ParameterSet.Config as cms

import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau1Regional = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau2Regional = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau3Regional = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
caloTowersTau4Regional = copy.deepcopy(caloTowerMakerHLT)
import copy
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
icone5Tau1Regional = copy.deepcopy(iterativeCone5CaloJets)
import copy
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
icone5Tau2Regional = copy.deepcopy(iterativeCone5CaloJets)
import copy
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
icone5Tau3Regional = copy.deepcopy(iterativeCone5CaloJets)
import copy
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
icone5Tau4Regional = copy.deepcopy(iterativeCone5CaloJets)
caloTausCreatorRegional = cms.Sequence(cms.SequencePlaceholder("doRegionalCaloForTaus")*caloTowersTau1Regional*icone5Tau1Regional*caloTowersTau2Regional*icone5Tau2Regional*caloTowersTau3Regional*icone5Tau3Regional*caloTowersTau4Regional*icone5Tau4Regional)
caloTowersTau1Regional.TauId = 0
caloTowersTau1Regional.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau1Regional.towers = 'towerMakerForTaus'
caloTowersTau2Regional.TauId = 1
caloTowersTau2Regional.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau2Regional.towers = 'towerMakerForTaus'
caloTowersTau3Regional.TauId = 2
caloTowersTau3Regional.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau3Regional.towers = 'towerMakerForTaus'
caloTowersTau4Regional.TauId = 3
caloTowersTau4Regional.TauTrigger = cms.InputTag("l1extraParticles","Tau")
caloTowersTau4Regional.towers = 'towerMakerForTaus'
icone5Tau1Regional.src = 'caloTowersTau1Regional'
icone5Tau2Regional.src = 'caloTowersTau2Regional'
icone5Tau3Regional.src = 'caloTowersTau3Regional'
icone5Tau4Regional.src = 'caloTowersTau4Regional'

