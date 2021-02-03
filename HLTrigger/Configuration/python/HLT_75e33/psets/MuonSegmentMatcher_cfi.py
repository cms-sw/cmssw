import FWCore.ParameterSet.Config as cms

MuonSegmentMatcher = cms.PSet(
    MatchParameters = cms.PSet(
        CSCsegments = cms.InputTag("cscSegments"),
        DTradius = cms.double(0.01),
        DTsegments = cms.InputTag("dt4DSegments"),
        RPChits = cms.InputTag("rpcRecHits"),
        TightMatchCSC = cms.bool(True),
        TightMatchDT = cms.bool(False)
    )
)