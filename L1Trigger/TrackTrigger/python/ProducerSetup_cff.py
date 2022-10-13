import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
from L1Trigger.TrackTrigger.ProducerSetup_cfi import TrackTrigger_params

dd4hep.toModify(TrackTrigger_params,
                fromDD4hep = cms.bool(True),
                ProcessHistory = cms.PSet (
                    GeometryConfiguration = cms.string("DDDetectorESProducer@"),
                    TTStubAlgorithm       = cms.string("TTStubAlgorithm_official_Phase2TrackerDigi_@")
                    )
)

TrackTriggerSetup = cms.ESProducer("tt::ProducerSetup", TrackTrigger_params)
