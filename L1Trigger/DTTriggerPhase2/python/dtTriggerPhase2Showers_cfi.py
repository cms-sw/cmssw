""" Basic config for running showers in Phase2 """
import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *

dtTriggerPhase2Shower = cms.EDProducer("DTTrigPhase2ShowerProd",
                                        digiTag = cms.InputTag("CalibratedDigis"),
                                        showerTaggingAlgo = cms.int32(1),
                                        threshold_for_shower = cms.int32(6),
                                        nHits_per_bx = cms.int32(8),
                                        obdt_hits_bxpersistence = cms.int32(4),
                                        obdt_wire_relaxing_time = cms.int32(2),
                                        bmtl1_hits_bxpersistence = cms.int32(16),
                                        debug = cms.untracked.bool(True),
                                        scenario = cms.int32(0))

