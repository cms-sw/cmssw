import FWCore.ParameterSet.Config as cms

print "\n!!!!! WARNING: The BMTF Packer returns the entire detector information in one fed (# 1376), instead of the real BMTF which owns 2 feds (1376, 1377)  !!!!!\n"


BMTFStage2Digis = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::BMTFSetup"),
    InputLabel = cms.InputTag("BMTFStage2Digis","BMTF"),
    InputLabel2 = cms.InputTag("BMTFStage2Digis"),
    FedId = cms.int32(1376),
    FWId = cms.uint32(1),
)
