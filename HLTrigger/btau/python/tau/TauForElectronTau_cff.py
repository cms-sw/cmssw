import FWCore.ParameterSet.Config as cms

import copy
from RecoTauTag.HLTProducers.L2TauJetsProvider_cfi import *
hltL2TauJetsProviderElectronTau = copy.deepcopy(l2TauJetsProvider)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
hltConeIsolationL25ElectronTau = copy.deepcopy(coneIsolationForHLT)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
hltConeIsolationL3ElectronTau = copy.deepcopy(coneIsolationForHLT)
import copy
from HLTrigger.btau.tau.JetCrystalsAssociator_cfi import *
hltJetCrystalsAssociatorElectronTau = copy.deepcopy(jetCrystalsAssociator)
import copy
from HLTrigger.btau.tau.EcalIsolation_cfi import *
hltEcalIsolationElectronTau = copy.deepcopy(ecalIsolation)
import copy
from RecoTauTag.HLTProducers.coneIsolationForHLT_cfi import *
hltPixelTrackConeIsolationElectronTau = copy.deepcopy(coneIsolationForHLT)
hltJetTracksAssociatorAtVertexL25ElectronTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2TauJetsProviderElectronTau"),
    tracks = cms.InputTag("pixelTracks"),
    coneSize = cms.double(0.5)
)

hltIsolatedTauJetsSelectorL25ElectronTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25ElectronTau")),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    UseVertex = cms.bool(False)
)

hltFilterIsolatedTauJetsL25ElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltJetTracksAssociatorAtVertexL3ElectronTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3ElectronTau"),
    coneSize = cms.double(0.5)
)

hltIsolatedTauJetsSelectorL3ElectronTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3ElectronTau")),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    UseVertex = cms.bool(False)
)

hltFilterIsolatedTauJetsL3ElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedTauJetsSelectorL3ElectronTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltEMIsolatedTauJetsSelectorElectronTau = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("hltEcalIsolationElectronTau"))
)

hltFilterEcalIsolatedTauJetsElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltEMIsolatedTauJetsSelectorElectronTau","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltJetsPixelTracksAssociatorElectronTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltEMIsolatedTauJetsSelectorElectronTau","Isolated"),
    tracks = cms.InputTag("pixelTracks"),
    coneSize = cms.double(0.5)
)

hltPixelTrackIsolatedTauJetsSelectorElectronTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("pixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltPixelTrackConeIsolationElectronTau")),
    IsolationCone = cms.double(0.3),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterPixelTrackIsolatedTauJetsElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1)
)

hltL2TauJetsProviderElectronTau.L1TauTrigger = 'hltLevel1GTSeedElectronTau'
hltL2TauJetsProviderElectronTau.JetSrc = ['icone5Tau1Regional', 'icone5Tau2Regional', 'icone5Tau3Regional', 'icone5Tau4Regional']
hltConeIsolationL25ElectronTau.JetTrackSrc = 'hltJetTracksAssociatorAtVertexL25ElectronTau'
hltConeIsolationL3ElectronTau.JetTrackSrc = 'hltJetTracksAssociatorAtVertexL3ElectronTau'
hltConeIsolationL3ElectronTau.useVertex = False
hltJetCrystalsAssociatorElectronTau.jets = 'hltL2TauJetsProviderElectronTau'
hltJetCrystalsAssociatorElectronTau.EBRecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEB")
hltJetCrystalsAssociatorElectronTau.EERecHits = cms.InputTag("ecalRecHitAll","EcalRecHitsEE")
hltEcalIsolationElectronTau.JetForFilter = 'hltJetCrystalsAssociatorElectronTau'
hltPixelTrackConeIsolationElectronTau.JetTrackSrc = 'hltJetsPixelTracksAssociatorElectronTau'
hltPixelTrackConeIsolationElectronTau.MinimumNumberOfHits = 2

