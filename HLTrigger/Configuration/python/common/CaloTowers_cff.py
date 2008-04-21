import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersES_cfi import *
import copy
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMakerForMuons = copy.deepcopy(towerMaker)
import copy
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMakerForTaus = copy.deepcopy(towerMaker)
import copy
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMakerForJets = copy.deepcopy(towerMaker)
import copy
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMakerForAll = copy.deepcopy(towerMaker)
caloTowersForMuons = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMakerForMuons"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

caloTowersForJets = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMakerForJets"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

caloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMakerForAll"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

doRegionalCaloForMuons = cms.Sequence(cms.SequencePlaceholder("doRegionalMuonsEcal")+cms.SequencePlaceholder("doLocalHcal")*towerMakerForMuons*caloTowersForMuons)
doRegionalCaloForTaus = cms.Sequence(cms.SequencePlaceholder("doRegionalTausEcal")+cms.SequencePlaceholder("doLocalHcal")*towerMakerForTaus)
doRegionalCaloForJets = cms.Sequence(cms.SequencePlaceholder("doRegionalJetsEcal")+cms.SequencePlaceholder("doLocalHcal")*towerMakerForJets*caloTowersForJets)
caloTowersRec = cms.Sequence(towerMakerForAll+caloTowers)
doCalo = cms.Sequence(cms.SequencePlaceholder("doEcalAll")+cms.SequencePlaceholder("doLocalHcal")+caloTowersRec)
towerMakerForMuons.ecalInputs = cms.VInputTag(cms.InputTag("ecalRegionalMuonsRecHit","EcalRecHitsEB"), cms.InputTag("ecalRegionalMuonsRecHit","EcalRecHitsEE"))
towerMakerForTaus.ecalInputs = cms.VInputTag(cms.InputTag("ecalRegionalTausRecHit","EcalRecHitsEB"), cms.InputTag("ecalRegionalTausRecHit","EcalRecHitsEE"))
towerMakerForJets.ecalInputs = cms.VInputTag(cms.InputTag("ecalRegionalJetsRecHit","EcalRecHitsEB"), cms.InputTag("ecalRegionalJetsRecHit","EcalRecHitsEE"))
towerMakerForAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHitAll","EcalRecHitsEB"), cms.InputTag("ecalRecHitAll","EcalRecHitsEE"))

