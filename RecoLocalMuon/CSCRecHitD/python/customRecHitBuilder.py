import FWCore.ParameterSet.Config as cms

def tightenAnodeTimes(process):
    if hasattr(process,'csc2DRecHits'):
        process.csc2DRecHits.CSCUseReducedWireTimeWindow = True
        process.csc2DRecHits.CSCWireTimeWindowLow = 5
        process.csc2DRecHits.CSCWireTimeWindowHigh = 11
        return process
