import FWCore.ParameterSet.Config as cms

GlobalMuonTrackMatcher = cms.PSet(
    GlobalMuonTrackMatcher = cms.PSet(
        Chi2Cut_1 = cms.double(50.0),
        Chi2Cut_2 = cms.double(50.0),
        Chi2Cut_3 = cms.double(200.0),
        DeltaDCut_1 = cms.double(2.5),
        DeltaDCut_2 = cms.double(10.0),
        DeltaDCut_3 = cms.double(15.0),
        DeltaRCut_1 = cms.double(0.1),
        DeltaRCut_2 = cms.double(0.2),
        DeltaRCut_3 = cms.double(1.0),
        Eta_threshold = cms.double(1.2),
        LocChi2Cut = cms.double(20.0),
        MinP = cms.double(2.5),
        MinPt = cms.double(1.0),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        Pt_threshold1 = cms.double(0.0),
        Pt_threshold2 = cms.double(999999999.0),
        Quality_1 = cms.double(20.0),
        Quality_2 = cms.double(15.0),
        Quality_3 = cms.double(7.0)
    )
)
