import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tStage2EmtfRatioClient = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TStage2EMTF/Timing/"),
    efficiency = cms.vstring(
        "cscLCTTimingFracBXNeg2 'CSC Chamber Occupancy BX -2' cscLCTTimingBXNeg2 cscTimingTotal",
        "cscLCTTimingFracBXNeg1 'CSC Chamber Occupancy BX -1' cscLCTTimingBXNeg1 cscTimingTotal",
        "cscLCTTimingFracBX0 'CSC Chamber Occupancy BX 0' cscLCTTimingBX0 cscTimingTotal",
        "cscLCTTimingFracBXPos1 'CSC Chamber Occupancy BX +1' cscLCTTimingBXPos1 cscTimingTotal",
        "cscLCTTimingFracBXPos2 'CSC Chamber Occupancy BX +2' cscLCTTimingBXPos2 cscTimingTotal",
        "rpcHitTimingFracBXNeg2 'RPC Chamber Occupancy BX -2' rpcHitTimingBXNeg2 rpcHitTimingTot",
        "rpcHitTimingFracBXNeg1 'RPC Chamber Occupancy BX -1' rpcHitTimingBXNeg1 rpcHitTimingTot",
        "rpcHitTimingFracBX0 'RPC Chamber Occupancy BX 0' rpcHitTimingBX0 rpcHitTimingTot",
        "rpcHitTimingFracBXPos1 'RPC Chamber Occupancy BX +1' rpcHitTimingBXPos1 rpcHitTimingTot",
        "rpcHitTimingFracBXPos2 'RPC Chamber Occupancy BX +2' rpcHitTimingBXPos2 rpcHitTimingTot",
        ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

# sequences
l1tStage2EmtfClient = cms.Sequence(
    l1tStage2EmtfRatioClient
)
