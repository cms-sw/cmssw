""" Basic config for running showers in Phase2 """
import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *

dtTriggerPhase2Shower = cms.EDProducer("DTTrigPhase2ShowerProd",
                                        digiTag = cms.InputTag("CalibratedDigis"),
                                        nHits_per_bx = cms.int32(3),
                                        threshold_for_shower = cms.int32(20),
                                        showerTaggingAlgo = cms.int32(1),
                                        debug = cms.untracked.bool(True))

