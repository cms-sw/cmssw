import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.ProducerES_cfi import TrackTrigger_params

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify(TrackTrigger_params,
                fromDD4hep = cms.bool(True),
                ProcessHistory = cms.PSet (
                    GeometryConfiguration = cms.string("DDDetectorESProducer@"),
                    TTStubAlgorithm       = cms.string("TTStubAlgorithm_official_Phase2TrackerDigi_@")
                    )
)

TrackTriggerSetup = cms.ESProducer("trackerDTC::ProducerES", TrackTrigger_params)
