import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
doubleTauPrescaler = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
doubleTauL1SeedFilter = copy.deepcopy(hltLevel1GTSeed)
import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
l2DoubleTauJets = copy.deepcopy(l2TauJetsProvider)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
doubleTauJetCrystalsAssociator = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
ecalDoubleTauIsol = copy.deepcopy(ecalIsolation)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
coneIsolationL25PixelTauIsolated = copy.deepcopy(coneIsolationForHLT)
import copy
from HLTrigger.btau.tau.TauHLTOpen_cfi import *
doubleTauHLTOpen = copy.deepcopy(hltTauProducer)
ecalDoubleTauIsolated = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("ecalDoubleTauIsol"))
)

filterDoubleTauEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("ecalDoubleTauIsolated","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(2)
)

associatorL25PixelTauIsolated = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("ecalDoubleTauIsolated","Isolated"),
    tracks = cms.InputTag("pixelTracks"),
    coneSize = cms.double(0.5)
)

isolatedL25PixelTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("coneIsolationL25PixelTauIsolated")),
    IsolationCone = cms.double(0.3),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

filterL25PixelTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("isolatedL25PixelTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2)
)

ecalIsolationDoubleTau = cms.Sequence(doubleTauJetCrystalsAssociator*ecalDoubleTauIsol*ecalDoubleTauIsolated)
ecalIsolatedPixelTauL25 = cms.Sequence(associatorL25PixelTauIsolated*coneIsolationL25PixelTauIsolated*isolatedL25PixelTau)
doubleCaloPixelTau = cms.Sequence(ecalIsolatedPixelTauL25+filterL25PixelTau)
doubleCaloPixelTauNoFilters = cms.Sequence(ecalIsolatedPixelTauL25)
doubleTauL1SeedFilter.L1SeedsLogicalExpression = 'L1_DoubleTauJet40'
l2DoubleTauJets.L1TauTrigger = 'doubleTauL1SeedFilter'
l2DoubleTauJets.JetSrc = ['icone5Tau1Regional', 'icone5Tau2Regional', 'icone5Tau3Regional', 'icone5Tau4Regional']
doubleTauJetCrystalsAssociator.jets = 'l2DoubleTauJets'
doubleTauJetCrystalsAssociator.EBRecHits = cms.InputTag("ecalRegionalTausRecHit","EcalRecHitsEB")
doubleTauJetCrystalsAssociator.EERecHits = cms.InputTag("ecalRegionalTausRecHit","EcalRecHitsEE")
ecalDoubleTauIsol.JetForFilter = 'doubleTauJetCrystalsAssociator'
coneIsolationL25PixelTauIsolated.JetTrackSrc = 'associatorL25PixelTauIsolated'
coneIsolationL25PixelTauIsolated.MinimumNumberOfHits = 2
coneIsolationL25PixelTauIsolated.MaximumTransverseImpactParameter = 1.
doubleTauHLTOpen.L2EcalIsoJets = 'ecalDoubleTauIsol'
doubleTauHLTOpen.L25TrackIsoJets = 'isolatedL25PixelTau'
doubleTauHLTOpen.L3TrackIsoJets = 'isolatedL25PixelTau'
doubleTauHLTOpen.SignalCone = 0.07
doubleTauHLTOpen.IsolationCone = 0.3

