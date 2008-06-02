import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.CommonModules_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
JpsitoMumuL1Seed = copy.deepcopy(hltLevel1GTSeed)
JpsitoMumuL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("JpsitoMumuL1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

JpsitoMumuL2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("JpsitoMumuL1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(-1),
    MaxInvMass = cms.double(10.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(2.0)
)

displacedJpsitoMumuFilter = cms.EDFilter("HLTDisplacedmumuFilter",
    Src = cms.InputTag("hltMuTracks"),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(4.0),
    ChargeOpt = cms.int32(-1),
    MaxEta = cms.double(2.5),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(6.0),
    MinPtPair = cms.double(4.0),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(1.0)
)

btoJpsitoMumu = cms.Sequence(cms.SequencePlaceholder("hltBegin")+JpsitoMumuL1Seed+JpsitoMumuL1Filtered+l2muonreco+JpsitoMumuL2Filtered+cms.SequencePlaceholder("l3displacedMumureco")+displacedJpsitoMumuFilter)
btoJpsitoMumunoL1emulation = cms.Sequence(JpsitoMumuL1Seed+JpsitoMumuL1Filtered+l2muonreco+JpsitoMumuL2Filtered+cms.SequencePlaceholder("l3displacedMumureco")+displacedJpsitoMumuFilter)
JpsitoMumuL1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

