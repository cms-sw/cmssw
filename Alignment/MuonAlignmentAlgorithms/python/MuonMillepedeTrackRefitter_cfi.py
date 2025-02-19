import FWCore.ParameterSet.Config as cms

MuonMillepedeTrackRefitter = cms.EDProducer("MuonMillepedeTrackRefitter",
    MuonCollectionTag = cms.InputTag("cosmicMuons"),
    TrackerTrackCollectionTag = cms.InputTag("cosmicMuonsTracker"),
    SATrackCollectionTag = cms.InputTag("cosmicMuonsStandAlone"),
    PropagatorSourceOpposite = cms.string('SmartPropagatorOpposite'),
    PropagatorSourceAlong = cms.string('SmartPropagator'),
    #PropagatorSourceOpposite = cms.string('SmartPropagatorAny'),
    #PropagatorSourceAlong = cms.string('SmartPropagatorAny'),
    SelectorOfFirstPoint = cms.int32(1),
    SegmentToTrackAssociatorParameters = cms.PSet(
      segmentsDT = cms.InputTag("dt4DSegments"),
      segmentsCSC = cms.InputTag("cscSegments")
    )
)


