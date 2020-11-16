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

## Era: Run2_2016
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(bmtfStage2Raw, InputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(1))

## Era: Run2_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify(bmtfStage2Raw, InputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(1))

### Era: Run2_2018
from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
stage2L1Trigger_2018.toModify(bmtfStage2Raw, InputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(1))

### Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(bmtfStage2Raw, InputLabel = cms.InputTag("simKBmtfDigis", "BMTF"), FWId = cms.uint32(2499805536))
