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

## FIXME: these are needed for ALCARECO production in CMSSW_14_0_X
## to avoid loosing in efficiency. To be reviewed after muon reco is fixed

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(TkAlGoodIdMuonSelector,
                       cut = '(abs(eta) < 2.5 & isGlobalMuon & isTrackerMuon & numberOfMatches > 1 & globalTrack.hitPattern.numberOfValidMuonHits > 0 &  globalTrack.normalizedChi2 < 20.) ||'  # regular selection
                       '(abs(eta) > 2.3 & abs(eta) < 3.0 & numberOfMatches >= 0 & isTrackerMuon)'   # to recover GE0 tracks
                       )

phase2_common.toModify(TkAlRelCombIsoMuonSelector,
                       cut = '(isolationR03().sumPt)/pt < 0.1' # only tracker isolation
                       )
