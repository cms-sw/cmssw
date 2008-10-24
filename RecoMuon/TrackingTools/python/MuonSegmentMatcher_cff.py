import FWCore.ParameterSet.Config as cms

MuonSegmentMatcher = cms.PSet(
    MatchParameters = cms.PSet(
        DTsegments = cms.untracked.InputTag("dt4DSegments"),
        CSCsegments = cms.untracked.InputTag("cscSegments"),
        TightMatchDT = cms.bool(False),
        TightMatchCSC = cms.bool(True)
    )
)