import FWCore.ParameterSet.Config as cms

TkAlGoodIdMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string('obj.isGlobalMuon() &&'
                     'obj.isTrackerMuon() &&'
                     'obj.numberOfMatches() > 1 &&'
                     'obj.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 &&'
                     'std::abs(obj.eta()) < 2.5 &&'
                     'obj.globalTrack()->normalizedChi2() < 20.'),
    filter = cms.bool(True)
)

TkAlRelCombIsoMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag(''),
    cut = cms.string('(obj.isolationR03().sumPt + obj.isolationR03().emEt + obj.isolationR03().hadEt)/obj.pt()  < 0.15'),
    filter = cms.bool(True)
)
