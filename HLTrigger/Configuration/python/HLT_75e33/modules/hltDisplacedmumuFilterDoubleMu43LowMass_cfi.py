import FWCore.ParameterSet.Config as cms

# Run-3 hltDisplacedmumuFilterDoubleMu43LowMass (HLT_GRun): vertex probability > 0.005, no Lxy or pointing-angle cut
hltDisplacedmumuFilterDoubleMu43LowMass = cms.EDFilter("HLTDisplacedmumuFilter",
    saveTags = cms.bool(True),
    FastAccept = cms.bool(True),
    MinLxySignificance = cms.double(0.0),
    MaxLxySignificance = cms.double(-1.0),
    MaxNormalisedChi2 = cms.double(999999.0),
    MinVtxProbability = cms.double(0.005),
    MinCosinePointingAngle = cms.double(-2.0),
    DisplacedVertexTag = cms.InputTag("hltDisplacedmumuVtxProducerDoubleMu43LowMass"),
    BeamSpotTag = cms.InputTag("hltOnlineBeamSpot"),
    MuonTag = cms.InputTag("hltPhase2L3MuonCandidates")
)
