import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
singleTauMETPrescaler = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
singleTauMETL1SeedFilter = copy.deepcopy(hltLevel1GTSeed)
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
l2SingleTauMETJets = copy.deepcopy(l2TauJetsProvider)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
singleTauMETJetCrystalsAssociator = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
ecalSingleTauMETIsol = copy.deepcopy(ecalIsolation)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
coneIsolationL25SingleTauMET = copy.deepcopy(coneIsolationForHLT)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
coneIsolationL3SingleTauMET = copy.deepcopy(coneIsolationForHLT)
import copy
from HLTrigger.btau.tau.TauHLTOpen_cfi import *
singleTauMETHLTOpen = copy.deepcopy(hltTauProducer)
hlt1METSingleTauMET = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("met"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(1)
)

ecalSingleTauMETIsolated = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("ecalSingleTauMETIsol"))
)

filterSingleTauMETEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("ecalSingleTauMETIsolated","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

associatorL25SingleTauMET = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("ecalSingleTauMETIsolated","Isolated"),
    tracks = cms.InputTag("ctfWithMaterialTracksL25SingleTauMET"),
    coneSize = cms.double(0.5)
)

isolatedL25SingleTauMET = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.065),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("coneIsolationL25SingleTauMET")),
    IsolationCone = cms.double(0.4),
    MinimumTransverseMomentumLeadingTrack = cms.double(15.0),
    UseVertex = cms.bool(False)
)

filterL25SingleTauMET = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("isolatedL25SingleTauMET"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1)
)

associatorL3SingleTauMET = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("isolatedL25SingleTauMET"),
    tracks = cms.InputTag("ctfWithMaterialTracksL3SingleTauMET"),
    coneSize = cms.double(0.5)
)

isolatedL3SingleTauMET = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.065),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("coneIsolationL3SingleTauMET")),
    IsolationCone = cms.double(0.4),
    MinimumTransverseMomentumLeadingTrack = cms.double(15.0),
    UseVertex = cms.bool(False)
)

filterL3SingleTauMET = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("isolatedL3SingleTauMET"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1)
)

ecalIsolationSingleTauMET = cms.Sequence(singleTauMETJetCrystalsAssociator*ecalSingleTauMETIsol*ecalSingleTauMETIsolated)
tauJetsMakerL25SingleTauMET = cms.Sequence(associatorL25SingleTauMET*coneIsolationL25SingleTauMET*isolatedL25SingleTauMET)
singleTauMETL25 = cms.Sequence(cms.SequencePlaceholder("ckfTracksL25SingleTauMET")*tauJetsMakerL25SingleTauMET)
tauJetsMakerL3SingleTauMET = cms.Sequence(associatorL3SingleTauMET*coneIsolationL3SingleTauMET*isolatedL3SingleTauMET)
singleTauMETL3 = cms.Sequence(cms.SequencePlaceholder("ckfTracksL3SingleTauMET")*tauJetsMakerL3SingleTauMET)
singleTauMET = cms.Sequence(singleTauMETL25+filterL25SingleTauMET+singleTauMETL3+filterL3SingleTauMET)
singleTauMETNoFilters = cms.Sequence(singleTauMETL25*singleTauMETL3)
singleTauMETL1SeedFilter.L1SeedsLogicalExpression = 'L1_TauJet30_ETM30'
l2SingleTauMETJets.L1TauTrigger = 'singleTauMETL1SeedFilter'
l2SingleTauMETJets.JetSrc = ['icone5Tau1', 'icone5Tau2', 'icone5Tau3', 'icone5Tau4']
singleTauMETJetCrystalsAssociator.jets = 'l2SingleTauMETJets'
singleTauMETJetCrystalsAssociator.EBRecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEB")
singleTauMETJetCrystalsAssociator.EERecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEE")
ecalSingleTauMETIsol.JetForFilter = 'singleTauMETJetCrystalsAssociator'
coneIsolationL25SingleTauMET.JetTrackSrc = 'associatorL25SingleTauMET'
coneIsolationL3SingleTauMET.JetTrackSrc = 'associatorL3SingleTauMET'
coneIsolationL3SingleTauMET.useVertex = True
singleTauMETHLTOpen.L2EcalIsoJets = 'ecalSingleTauMETIsol'
singleTauMETHLTOpen.L25TrackIsoJets = 'isolatedL25SingleTauMET'
singleTauMETHLTOpen.L3TrackIsoJets = 'isolatedL3SingleTauMET'
singleTauMETHLTOpen.SignalCone = 0.065
singleTauMETHLTOpen.IsolationCone = 0.4

