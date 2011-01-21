import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

# My New Producer
interestingTrackEcalDetIdProducer = cms.EDProducer('InterestingTrackEcalDetIdProducer',
    TrackAssociatorParameterBlock,
    TrackCollection = cms.InputTag("generalTracks"),
    MinTrackPt      = cms.double(50.0)
)

