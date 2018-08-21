from __future__ import print_function
import FWCore.ParameterSet.Config as cms


bmtfStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::BMTFSetup"),
    InputLabel = cms.InputTag("simBmtfDigis","BMTF"),
    InputLabel2 = cms.InputTag("simTwinMuxDigis"),
    FedId = cms.int32(1376),
    FWId = cms.uint32(1),
)
