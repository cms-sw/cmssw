import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
singleTauPrescaler = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
singleTauL1SeedFilter = copy.deepcopy(hltLevel1GTSeed)
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
l2SingleTauJets = copy.deepcopy(l2TauJetsProvider)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
singleTauJetCrystalsAssociator = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
ecalSingleTauIsol = copy.deepcopy(ecalIsolation)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
coneIsolationL25SingleTau = copy.deepcopy(coneIsolationForHLT)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
coneIsolationL3SingleTau = copy.deepcopy(coneIsolationForHLT)
import copy
from HLTrigger.btau.tau.TauHLTOpen_cfi import *
singleTauHLTOpen = copy.deepcopy(hltTauProducer)
hlt1METSingleTau = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("met"),
    MinPt = cms.double(65.0),
    MinN = cms.int32(1)
)

ecalSingleTauIsolated = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("ecalSingleTauIsol"))
)

filterSingleTauEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("ecalSingleTauIsolated","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

associatorL25SingleTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("ecalSingleTauIsolated","Isolated"),
    tracks = cms.InputTag("ctfWithMaterialTracksL25SingleTau"),
    coneSize = cms.double(0.5)
)

isolatedL25SingleTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.065),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("coneIsolationL25SingleTau")),
    IsolationCone = cms.double(0.4),
    MinimumTransverseMomentumLeadingTrack = cms.double(20.0),
    UseVertex = cms.bool(False)
)

filterL25SingleTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("isolatedL25SingleTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

associatorL3SingleTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("isolatedL25SingleTau"),
    tracks = cms.InputTag("ctfWithMaterialTracksL3SingleTau"),
    coneSize = cms.double(0.5)
)

isolatedL3SingleTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.065),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("coneIsolationL3SingleTau")),
    IsolationCone = cms.double(0.4),
    MinimumTransverseMomentumLeadingTrack = cms.double(20.0),
    UseVertex = cms.bool(False)
)

filterL3SingleTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("isolatedL3SingleTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

tauJetsMakerL25SingleTau = cms.Sequence(associatorL25SingleTau*coneIsolationL25SingleTau*isolatedL25SingleTau)
singleTauL25 = cms.Sequence(cms.SequencePlaceholder("ckfTracksL25SingleTau")*tauJetsMakerL25SingleTau)
tauJetsMakerL3SingleTau = cms.Sequence(associatorL3SingleTau*coneIsolationL3SingleTau*isolatedL3SingleTau)
singleTauL3 = cms.Sequence(cms.SequencePlaceholder("ckfTracksL3SingleTau")*tauJetsMakerL3SingleTau)
ecalIsolationSingleTau = cms.Sequence(singleTauJetCrystalsAssociator*ecalSingleTauIsol*ecalSingleTauIsolated)
singleTau = cms.Sequence(singleTauL25+filterL25SingleTau+singleTauL3+filterL3SingleTau)
singleTauNoFilters = cms.Sequence(singleTauL25+singleTauL3)
singleTauL1SeedFilter.L1SeedsLogicalExpression = 'L1_SingleTauJet80'
l2SingleTauJets.L1TauTrigger = 'singleTauL1SeedFilter'
l2SingleTauJets.JetSrc = ['icone5Tau1', 'icone5Tau2', 'icone5Tau3', 'icone5Tau4']
singleTauJetCrystalsAssociator.jets = 'l2SingleTauJets'
singleTauJetCrystalsAssociator.EBRecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEB")
singleTauJetCrystalsAssociator.EERecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEE")
ecalSingleTauIsol.JetForFilter = 'singleTauJetCrystalsAssociator'
coneIsolationL25SingleTau.JetTrackSrc = 'associatorL25SingleTau'
coneIsolationL3SingleTau.JetTrackSrc = 'associatorL3SingleTau'
coneIsolationL3SingleTau.useVertex = True
singleTauHLTOpen.L2EcalIsoJets = 'ecalSingleTauIsol'
singleTauHLTOpen.L25TrackIsoJets = 'isolatedL25SingleTau'
singleTauHLTOpen.SignalCone = 0.065
singleTauHLTOpen.IsolationCone = 0.4
singleTauHLTOpen.L3TrackIsoJets = 'isolatedL3SingleTau'

