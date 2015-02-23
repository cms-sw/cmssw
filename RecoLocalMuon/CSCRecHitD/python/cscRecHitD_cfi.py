import FWCore.ParameterSet.Config as cms

# This object modifies the csc2DRecHits for running in Run 2
from Configuration.StandardSequences.Eras import eras

# parameters for CSC rechit building
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cff import *
csc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    #
    #    Parameters for coordinate and uncertainty calculations
    #    Data and MC parameters are (still) different
    #    Needs tuning
    #
    cscRecHitDParameters,
    #
    #    Parameters for strip hits
    #
    CSCStripPeakThreshold = cms.double(10.0),
    CSCStripClusterChargeCut = cms.double(25.0),
    CSCStripxtalksOffset = cms.double(0.03),
    #
    #    How to find SCA peak time?
    #                              
    UseAverageTime = cms.bool(False),
    UseParabolaFit = cms.bool(False),
    UseFivePoleFit = cms.bool(True),                       
    #
    #    Parameters for wire hits
    CSCWireClusterDeltaT = cms.int32(1),
    #
    #    Calibration info:
    CSCUseCalibrations = cms.bool(True),
    #    Pedestal treatment
    CSCUseStaticPedestals = cms.bool(False),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32(2),
    #
    #    Which digis:
    #
    #  When using data from unpacker
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    #  When using data from simulation
    #    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    #    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    #
    # Use dead channels information 
    readBadChannels = cms.bool(True),
    readBadChambers = cms.bool(True),
    #                          
    # Do we use the chip and chamber and L1A phase corrections when filling the recHit time?
    #
    CSCUseTimingCorrections = cms.bool(True),
    #
    # Do we correct the energy deposited for gas gains?
    CSCUseGasGainCorrections = cms.bool(True),
    #
    #    Parameters which are not used currently
    #
    CSCDebug = cms.untracked.bool(False),
    #  To be set once wire digis have proper timing info:
    CSCstripWireDeltaTime = cms.int32(8),
    # to be deleted
    CSCStripClusterSize = cms.untracked.int32(3)
)

##
## Modify for running in Run 2
##
eras.run2_common.toModify( csc2DRecHits, readBadChannels = False )
eras.run2_common.toModify( csc2DRecHits, CSCUseGasGainCorrections = False )

