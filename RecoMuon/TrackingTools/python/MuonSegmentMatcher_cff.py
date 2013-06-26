import FWCore.ParameterSet.Config as cms

MuonSegmentMatcher = cms.PSet(
    MatchParameters = cms.PSet(
        DTsegments = cms.InputTag("dt4DSegments"),
        DTradius = cms.double(0.01),
        CSCsegments = cms.InputTag("cscSegments"),
        TightMatchDT = cms.bool(False),
        TightMatchCSC = cms.bool(True)
    )
)