import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.CommonModules_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
MuMukL1Seed = copy.deepcopy(hltLevel1GTSeed)
MuMukL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("MuMukL1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

MuMukL2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("JpsitoMumuL1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(1000.0),
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
    MinInvMass = cms.double(0.02),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(2.0)
)

displacedMuMukFilter = cms.EDFilter("HLTDisplacedmumuFilter",
    Src = cms.InputTag("hltMuTracks"),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(3.0),
    ChargeOpt = cms.int32(0),
    MaxEta = cms.double(2.5),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(3.0),
    MinPtPair = cms.double(0.0),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(0.2)
)

hltmmkFilter = cms.EDFilter("HLTmmkFilter",
    MinCosinePointingAngle = cms.double(0.9),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(3.0),
    MaxEta = cms.double(2.5),
    ThirdTrackMass = cms.double(0.106),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(2.2),
    TrackCand = cms.InputTag("hltMumukAllConeTracks"),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(1.2),
    MuCand = cms.InputTag("hltMuTracks")
)

Mumuk = cms.Sequence(cms.SequencePlaceholder("hltBegin")+MuMukL1Seed+MuMukL1Filtered+l2muonreco+MuMukL2Filtered+cms.SequencePlaceholder("l3displacedMumureco")+displacedMuMukFilter)
MumuknoL1emulation = cms.Sequence(MuMukL1Seed+MuMukL1Filtered+l2muonreco+MuMukL2Filtered+cms.SequencePlaceholder("l3displacedMumureco")+displacedMuMukFilter)
BToMuMuK = cms.Sequence(Mumuk+cms.SequencePlaceholder("l3MumukReco")+hltmmkFilter)
MuMukL1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

