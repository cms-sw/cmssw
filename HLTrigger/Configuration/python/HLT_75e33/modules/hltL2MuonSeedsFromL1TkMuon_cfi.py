import FWCore.ParameterSet.Config as cms

hltL2MuonSeedsFromL1TkMuon = cms.EDProducer("L2MuonSeedGeneratorFromL1TkMu",
    EtaMatchingBins = cms.vdouble(0.0, 2.5),
    InputObjects = cms.InputTag("l1tTkMuonsGmt"),
    L1MaxEta = cms.double(2.5),
    L1MinPt = cms.double(0.0),
    MatchDR = cms.vdouble(0.3),
    MinPL1Tk = cms.double(3.5),
    MinPtL1TkBarrel = cms.double(3.5),
    OfflineSeedLabel = cms.untracked.InputTag("hltL2OfflineMuonSeeds"),
    Propagator = cms.string('SteppingHelixPropagatorAny'),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    SetMinPtBarrelTo = cms.double(3.5),
    SetMinPtEndcapTo = cms.double(1.0),
    UseOfflineSeed = cms.untracked.bool(True),
    UseUnassociatedL1 = cms.bool(False)
)

phase2HltL2MuonSeedsFromL1TkMuon = cms.EDProducer('Phase2L2MuonSeedCreator',
    inputObjects = cms.InputTag('l1tTkMuonsGmt'),
    cscRecSegmentLabel = cms.InputTag('hltCscSegments'),
    dtRecSegmentLabel = cms.InputTag('hltDt4DSegments'),
    minPL1Tk = cms.double(3.5),
    maxPL1Tk = cms.double(200),
    stubMatchDPhi = cms.double(0.05),
    stubMatchDTheta = cms.double(0.1),
    extrapolationWindowClose = cms.double(0.2),
    extrapolationWindowFar = cms.double(0.1),
    maximumEtaBarrel = cms.double(0.7),
    maximumEtaOverlap = cms.double(1.3),
    propagator = cms.string('SteppingHelixPropagatorAny'),
    serviceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    estimatorMaxChi2 = cms.double(1000.0)
)   

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
phase2L2AndL3Muons.toReplaceWith(
    hltL2MuonSeedsFromL1TkMuon,
    phase2HltL2MuonSeedsFromL1TkMuon
    )
