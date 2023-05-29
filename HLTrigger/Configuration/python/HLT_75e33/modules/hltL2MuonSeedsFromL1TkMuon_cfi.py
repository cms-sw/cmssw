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
