import FWCore.ParameterSet.Config as cms

pfDisplacedTrackerVertex = cms.EDProducer("PFDisplacedTrackerVertexProducer",
                           # cut on the likelihood of the nuclear interaction
               displacedTrackerVertexColl = cms.InputTag("particleFlowDisplacedVertex"),
               trackColl = cms.InputTag("generalTracks")

                           )


