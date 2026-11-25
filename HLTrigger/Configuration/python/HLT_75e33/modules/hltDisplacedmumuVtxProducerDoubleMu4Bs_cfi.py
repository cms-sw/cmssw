import FWCore.ParameterSet.Config as cms

# Run-3 hltDisplacedmumuVtxProducerDoubleMu4Bs (HLT_GRun), on the Phase-2 L3 muon candidates
hltDisplacedmumuVtxProducerDoubleMu4Bs = cms.EDProducer("HLTDisplacedmumuVtxProducer",
    Src = cms.InputTag("hltPhase2L3MuonCandidates"),
    PreviousCandTag = cms.InputTag("hltDoubleMu4BsL3Filtered"),
    matchToPrevious = cms.bool(True),
    MaxEta = cms.double(2.5),
    MinPt = cms.double(0.0),
    MinPtPair = cms.double(0.0),
    MinInvMass = cms.double(0.0),
    MaxInvMass = cms.double(999999.0),
    ChargeOpt = cms.int32(-1)
)
