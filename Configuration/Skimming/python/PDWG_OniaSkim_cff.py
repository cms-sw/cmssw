import FWCore.ParameterSet.Config as cms


OniaSkimGoodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isGlobalMuon || (isTrackerMuon && numberOfMatches('SegmentAndTrackArbitration')>0)"),
)
OniaSkimDiMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("OniaSkimGoodMuons OniaSkimGoodMuons"),
    checkCharge = cms.bool(False),
    cut         = cms.string("mass > 2"),
)
OniaSkimDiMuonFilter = cms.EDFilter("CandViewCountFilter",
    src       = cms.InputTag("OniaSkimDiMuons"),
    minNumber = cms.uint32(1),
)

oniaSkimSequence = cms.Sequence(
    OniaSkimGoodMuons *
    OniaSkimDiMuons *
    OniaSkimDiMuonFilter
    )
