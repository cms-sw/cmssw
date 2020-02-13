import FWCore.ParameterSet.Config as cms

def tightenAnodeTimes(process):
    if hasattr(process,'csc2DRecHits'):
        process.csc2DRecHits.CSCUseReducedWireTimeWindow = cms.bool(True)
        process.csc2DRecHits.CSCWireTimeWindowLow = cms.int32(5)
        process.csc2DRecHits.CSCWireTimeWindowHigh = cms.int32(11)
        return process
