import FWCore.ParameterSet.Config as cms

GlobalMuonTrackMatcher = cms.PSet(
  GlobalMuonTrackMatcher = cms.PSet(
    MinP = cms.double(2.5),
    MinPt = cms.double(1.0),
    Pt_threshold= cms.double(35.0),
    Eta_threshold= cms.double(1.0),
    Chi2Cut_1= cms.double(30.0),
    Chi2Cut_2= cms.double(80.0),
    Chi2Cut_3= cms.double(200.0),
    LocChi2Cut= cms.double(.008),
    DeltaDCut_1= cms.double(20.0),
    DeltaDCut_2= cms.double(15.0),
    DeltaDCut_3= cms.double(30.0),
    DeltaRCut_1= cms.double(.1),
    DeltaRCut_2= cms.double(.15),
    DeltaRCut_3= cms.double(.20),
    Propagator = cms.string('SteppingHelixPropagatorAny')
  )
) 
