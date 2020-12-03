import FWCore.ParameterSet.Config as cms

bmtfDigis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::BMTFSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32(1376,1377),
    FWId = cms.uint32(1),
    FWOverride = cms.bool(False),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8),
    lenAMCHeader = cms.untracked.int32(8),
    lenAMCTrailer = cms.untracked.int32(0),
    lenAMC13Header = cms.untracked.int32(8),
    lenAMC13Trailer = cms.untracked.int32(8)
)

## Era: Run2_2016
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(bmtfDigis, FWId = cms.uint32(0x93500160))

## Era: Run2_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify(bmtfDigis, FWId = cms.uint32(0x93500160))

### Era: Run2_2018
from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
stage2L1Trigger_2018.toModify(bmtfDigis, FWId = cms.uint32(0x93500160))

### Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(bmtfDigis, FWId = cms.uint32(0x95030160))
