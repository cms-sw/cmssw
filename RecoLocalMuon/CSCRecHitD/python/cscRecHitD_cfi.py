import FWCore.ParameterSet.Config as cms

# This object modifies the csc2DRecHits for running in Run 2

# parameters for CSC rechit building
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cff import *
import RecoLocalMuon.CSCRecHitD.configWireTimeWindow_cfi as _mod

csc2DRecHits = _mod.configWireTimeWindow.clone(
    #    wire time window used for reconstruction
    CSCUseReducedWireTimeWindow = True,
    CSCWireTimeWindowLow = 5,
    CSCWireTimeWindowHigh = 11
    #
    #    Which digis:
    #
    #  When using data from simulation
    #    wireDigiTag  = "simMuonCSCDigis:MuonCSCWireDigi",
    #    stripDigiTag = "simMuonCSCDigis:MuonCSCStripDigi",
)

##
## Modify for running in Run 2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( csc2DRecHits, 
     readBadChannels = False, 
     CSCUseGasGainCorrections = False )
