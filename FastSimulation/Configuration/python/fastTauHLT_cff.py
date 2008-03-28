# The following comments couldn't be translated into the new config version:

#EcalIsolation Selectors

import FWCore.ParameterSet.Config as cms

#include files from Full simulation
from HLTrigger.btau.tau.SingleTau_cff import *
from HLTrigger.btau.tau.SingleTauMET_cff import *
from HLTrigger.btau.tau.DoubleTau_cff import *
#PixelTracks and Vertices
from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
import copy
from RecoTauTag.HLTProducers.caloTowerMakerHLT_cfi import *
# prepare tower collection in the location of the L1 Tau candidates
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
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
#L2TauJets provider (can be removed if fast simulation switches to the new L1TriggerEmulationMap)
l2SingleTauJetsFast = copy.deepcopy(l2TauJetsProvider)
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
l2SingleTauMETJetsFast = copy.deepcopy(l2TauJetsProvider)
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
l2DoubleTauJetsFast = copy.deepcopy(l2TauJetsProvider)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
#Replace needed for EcalIsolation
#construct the jet-crystals associator for the single and double Tau collection
singleTauJetCrystalsAssociatorFast = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
singleTauMETJetCrystalsAssociatorFast = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
doubleTauJetCrystalsAssociatorFast = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
#EcalIsolation Producer
ecalSingleTauIsolFast = copy.deepcopy(ecalIsolation)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
ecalSingleTauMETIsolFast = copy.deepcopy(ecalIsolation)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
ecalDoubleTauIsolFast = copy.deepcopy(ecalIsolation)
#L1 SeedFilter not anylonger in use in Full Simulation
singleTauL1SeedFilterFast = cms.EDFilter("HLTLevel1Seed",
    andOr = cms.bool(True),
    byName = cms.bool(True),
    L1ExtraParticleMap = cms.InputTag("fastL1extraParticleMap"),
    L1GTReadoutRecord = cms.InputTag("fastL1extraParticleMap"),
    L1Seeds = cms.vstring('L1_SingleTauJet80'),
    L1ExtraCollections = cms.InputTag("fastL1CaloSim")
)

singleTauMETL1SeedFilterFast = cms.EDFilter("HLTLevel1Seed",
    andOr = cms.bool(True),
    byName = cms.bool(True),
    L1ExtraParticleMap = cms.InputTag("fastL1extraParticleMap"),
    L1GTReadoutRecord = cms.InputTag("fastL1extraParticleMap"),
    L1Seeds = cms.vstring('L1_TauJet30_ETM30'),
    L1ExtraCollections = cms.InputTag("fastL1CaloSim")
)

doubleTauL1SeedFilterFast = cms.EDFilter("HLTLevel1Seed",
    andOr = cms.bool(True),
    byName = cms.bool(True),
    L1ExtraParticleMap = cms.InputTag("fastL1extraParticleMap"),
    L1GTReadoutRecord = cms.InputTag("fastL1extraParticleMap"),
    L1Seeds = cms.vstring('L1_DoubleTauJet40'),
    L1ExtraCollections = cms.InputTag("fastL1CaloSim")
)

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

#All the rest is the same!
#Sequences
caloTausCreator = cms.Sequence(caloTowersTau1*icone5Tau1*caloTowersTau2*icone5Tau2*caloTowersTau3*icone5Tau3*caloTowersTau4*icone5Tau4)
fastSingleTauTrigger = cms.Sequence(singleTauL1SeedFilterFast+caloTausCreator+cms.SequencePlaceholder("met")+hlt1METSingleTau+l2SingleTauJetsFast+singleTauJetCrystalsAssociatorFast+ecalSingleTauIsolFast+ecalSingleTauIsolated+filterSingleTauEcalIsolation+cms.SequencePlaceholder("famosWithTracks")+pixelGSTracking+pixelGSVertexing+associatorL3SingleTau*coneIsolationL3SingleTau*isolatedL3SingleTau*filterL3SingleTau)
fastSingleTauMETTrigger = cms.Sequence(singleTauMETL1SeedFilterFast+caloTausCreator+cms.SequencePlaceholder("met")+hlt1METSingleTauMET+l2SingleTauMETJetsFast+singleTauMETJetCrystalsAssociatorFast+ecalSingleTauMETIsolFast+ecalSingleTauMETIsolated+filterSingleTauMETEcalIsolation+cms.SequencePlaceholder("famosWithTracks")+pixelGSTracking+pixelGSVertexing+associatorL3SingleTauMET*coneIsolationL3SingleTauMET*isolatedL3SingleTauMET*filterL3SingleTauMET)
fastDoubleTauTrigger = cms.Sequence(doubleTauL1SeedFilterFast+caloTausCreator+l2DoubleTauJetsFast+doubleTauJetCrystalsAssociatorFast+ecalDoubleTauIsolFast+ecalDoubleTauIsolated+filterDoubleTauEcalIsolation+cms.SequencePlaceholder("famosWithTrackerHits")+pixelGSTracking+pixelGSVertexing+associatorL25PixelTauIsolated*coneIsolationL25PixelTauIsolated*isolatedL25PixelTau*filterL25PixelTau)
caloTowersTau1.TauId = 0
caloTowersTau1.TauTrigger = cms.InputTag("fastL1CaloSim","Tau")
caloTowersTau2.TauId = 1
caloTowersTau2.TauTrigger = cms.InputTag("fastL1CaloSim","Tau")
caloTowersTau3.TauId = 2
caloTowersTau3.TauTrigger = cms.InputTag("fastL1CaloSim","Tau")
caloTowersTau4.TauId = 3
caloTowersTau4.TauTrigger = cms.InputTag("fastL1CaloSim","Tau")
icone5Tau1.src = 'caloTowersTau1'
icone5Tau2.src = 'caloTowersTau2'
icone5Tau3.src = 'caloTowersTau3'
icone5Tau4.src = 'caloTowersTau4'
l2SingleTauJetsFast.L1TauTrigger = 'singleTauL1SeedFilterFast'
l2SingleTauJetsFast.L1Particles = cms.InputTag("fastL1CaloSim","Tau")
l2SingleTauMETJetsFast.L1TauTrigger = 'singleTauMETL1SeedFilterFast'
l2SingleTauMETJetsFast.L1Particles = cms.InputTag("fastL1CaloSim","Tau")
l2DoubleTauJetsFast.L1TauTrigger = 'doubleTauL1SeedFilterFast'
l2DoubleTauJetsFast.L1Particles = cms.InputTag("fastL1CaloSim","Tau")
singleTauJetCrystalsAssociatorFast.jets = 'l2SingleTauJetsFast'
singleTauJetCrystalsAssociatorFast.EBRecHits = cms.InputTag("caloRecHits","EcalRecHitsEB")
singleTauJetCrystalsAssociatorFast.EERecHits = cms.InputTag("caloRecHits","EcalRecHitsEE")
singleTauMETJetCrystalsAssociatorFast.jets = 'l2SingleTauMETJetsFast'
singleTauMETJetCrystalsAssociatorFast.EBRecHits = cms.InputTag("caloRecHits","EcalRecHitsEB")
singleTauMETJetCrystalsAssociatorFast.EERecHits = cms.InputTag("caloRecHits","EcalRecHitsEE")
doubleTauJetCrystalsAssociatorFast.jets = 'l2DoubleTauJetsFast'
doubleTauJetCrystalsAssociatorFast.EBRecHits = cms.InputTag("caloRecHits","EcalRecHitsEB")
doubleTauJetCrystalsAssociatorFast.EERecHits = cms.InputTag("caloRecHits","EcalRecHitsEE")
ecalSingleTauIsolFast.JetForFilter = 'singleTauJetCrystalsAssociatorFast'
ecalSingleTauMETIsolFast.JetForFilter = 'singleTauMETJetCrystalsAssociatorFast'
ecalDoubleTauIsolFast.JetForFilter = 'doubleTauJetCrystalsAssociatorFast'
ecalSingleTauIsolated.TauSrc = ['ecalSingleTauIsolFast']
ecalSingleTauMETIsolated.TauSrc = ['ecalSingleTauMETIsolFast']
ecalDoubleTauIsolated.TauSrc = ['ecalDoubleTauIsolFast']
#TrackIsolation modules
#JetTrackAssociator
associatorL3SingleTau.tracks = 'ctfGSWithMaterialTracks'
associatorL3SingleTau.jets = cms.InputTag("ecalSingleTauIsolated","Isolated")
associatorL3SingleTauMET.tracks = 'ctfGSWithMaterialTracks'
associatorL3SingleTauMET.jets = cms.InputTag("ecalSingleTauMETIsolated","Isolated")
associatorL25PixelTauIsolated.tracks = 'pixelGSTracks'
associatorL25PixelTauIsolated.jets = cms.InputTag("ecalDoubleTauIsolated","Isolated")

