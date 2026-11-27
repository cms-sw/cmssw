import FWCore.ParameterSet.Config as cms

hltL2MuonSeedsFromL1TkMuon = cms.EDProducer('Phase2L2MuonSeedCreator',
    inputObjects = cms.InputTag('l1tTkMuonsGmt'),
    cscRecSegmentLabel = cms.InputTag('hltCscSegments'),
    dtRecSegmentLabel = cms.InputTag('hltDt4DSegments'),
    minL1pT = cms.double(0.0),
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
