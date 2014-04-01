import FWCore.ParameterSet.Config as cms

TrajectoryFilterForConversions = cms.PSet(
    chargeSignificance = cms.double(-1.0),
    minPt = cms.double(0.9),
    minHitsMinPt = cms.int32(-1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(-1),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

