import FWCore.ParameterSet.Config as cms

import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
hltL2TauJetsProviderMuonTau = copy.deepcopy(l2TauJetsProvider)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
hltJetCrystalsAssociatorMuonTau = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
hltEcalIsolationMuonTau = copy.deepcopy(ecalIsolation)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
hltPixelTrackConeIsolationMuonTau = copy.deepcopy(coneIsolationForHLT)
hltEMIsolatedTauJetsSelectorMuonTau = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("hltEcalIsolationMuonTau"))
)

hltFilterEcalIsolatedTauJetsMuonTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltEMIsolatedTauJetsSelectorMuonTau","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltJetsPixelTracksAssociatorMuonTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltEMIsolatedTauJetsSelectorMuonTau","Isolated"),
    tracks = cms.InputTag("pixelTracks"),
    coneSize = cms.double(0.5)
)

hltPixelTrackIsolatedTauJetsSelectorMuonTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltPixelTrackConeIsolationMuonTau")),
    IsolationCone = cms.double(0.3),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterPixelTrackIsolatedTauJetsMuonTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1)
)

hltL2TauJetsProviderMuonTau.L1TauTrigger = 'hltLevel1GTSeedMuonTau'
hltL2TauJetsProviderMuonTau.JetSrc = ['icone5Tau1Regional', 'icone5Tau2Regional', 'icone5Tau3Regional', 'icone5Tau4Regional']
hltJetCrystalsAssociatorMuonTau.jets = 'hltL2TauJetsProviderMuonTau'
hltJetCrystalsAssociatorMuonTau.EBRecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEB")
hltJetCrystalsAssociatorMuonTau.EERecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEE")
hltEcalIsolationMuonTau.JetForFilter = 'hltJetCrystalsAssociatorMuonTau'
hltPixelTrackConeIsolationMuonTau.JetTrackSrc = 'hltJetsPixelTracksAssociatorMuonTau'
hltPixelTrackConeIsolationMuonTau.MinimumNumberOfHits = 2
hltPixelTrackConeIsolationMuonTau.MinimumTransverseMomentumLeadingTrack = 3.
hltPixelTrackConeIsolationMuonTau.useVertex = True

