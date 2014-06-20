import FWCore.ParameterSet.Config as cms

TkAlGoodIdMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string('isGlobalMuon &'
                     'isTrackerMuon &'
                     'numberOfMatches > 1 &'
                     'globalTrack.hitPattern.numberOfValidMuonHits > 0 &'
                     'abs(eta) < 2.5 &'
                     'globalTrack.normalizedChi2 < 20.'),
    filter = cms.bool(True)
)

TkAlRelCombIsoMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag(''),
    cut = cms.string('(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.15'),
    filter = cms.bool(True)
)
