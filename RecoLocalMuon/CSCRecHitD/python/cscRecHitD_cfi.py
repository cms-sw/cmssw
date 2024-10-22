import FWCore.ParameterSet.Config as cms

# This object modifies the csc2DRecHits for running in Run 2

# parameters for CSC rechit building
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cff import *
import RecoLocalMuon.CSCRecHitD.cscRecHitDProducer_cfi as _mod

csc2DRecHits = _mod.cscRecHitDProducer.clone(
    #
    #    Parameters for strip hits
    #
    CSCStripPeakThreshold = 10.0,
    CSCStripClusterChargeCut = 25.0,
    CSCStripxtalksOffset = 0.03,
    #
    #    How to find SCA peak time?
    #                              
    UseAverageTime = False,
    UseParabolaFit = False,
    UseFivePoleFit = True,                       
    #
    #    Parameters for wire hits
    CSCWireClusterDeltaT = 1,
    #
    #    wire time window used for reconstruction
    CSCUseReducedWireTimeWindow = True,
    CSCWireTimeWindowLow = 5,
    CSCWireTimeWindowHigh = 11,
    #
    #    Calibration info:
    CSCUseCalibrations = True,
    #    Pedestal treatment
    CSCUseStaticPedestals = False,
    CSCNoOfTimeBinsForDynamicPedestal = 2,
    #
    #    Which digis:
    #
    #  When using data from unpacker
    wireDigiTag = "muonCSCDigis:MuonCSCWireDigi",
    stripDigiTag = "muonCSCDigis:MuonCSCStripDigi",
    #  When using data from simulation
    #    wireDigiTag = "simMuonCSCDigis:MuonCSCWireDigi",
    #    stripDigiTag = "simMuonCSCDigis:MuonCSCStripDigi",
    #
    # Use dead channels information 
    readBadChannels = True,
    readBadChambers = True,
    #                          
    # Do we use the chip and chamber and L1A phase corrections when filling the recHit time?
    #
    CSCUseTimingCorrections = True,
    #
    # Do we correct the energy deposited for gas gains?
    CSCUseGasGainCorrections = True,
    #
    #    Parameters which are not used currently
    #
    CSCDebug = False,
    #  To be set once wire digis have proper timing info:
    CSCstripWireDeltaTime = 8,
    #
    #    Parameters for coordinate and uncertainty calculations
    #    Data and MC parameters are (still) different
    #    Needs tuning
    #
    **cscRecHitDParameters
)
##
## Modify for running in Run 2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( csc2DRecHits, 
     readBadChannels = False,
     CSCUseGasGainCorrections = False )
