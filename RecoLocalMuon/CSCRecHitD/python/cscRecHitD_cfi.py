import FWCore.ParameterSet.Config as cms

# parameters for CSC rechit building
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cff import *
csc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    #
    #    Parameters for coordinate and uncertainty calculations
    #    Data and MC parameters are (still) different
    #    Use cscRecHitD_data.cff if you run on data  
    #
    cscRecHitDParameters,
    CSCStripClusterSize = cms.untracked.int32(3),
    #
    #    Parameters for strip hits
    #
    CSCStripPeakThreshold = cms.untracked.double(10.0),
    #
    #    Parameters for wire hits
    CSCWireClusterDeltaT = cms.untracked.int32(1),
    CSCStripxtalksOffset = cms.untracked.double(0.03),
    #  To be set once wire digis have timing info:
    CSCstripWireDeltaTime = cms.untracked.int32(8),
    #
    #    Calibration info:
    CSCUseCalibrations = cms.untracked.bool(True),
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
    #    Parameters for 2-D hits (not used currently)
    #
    CSCDebug = cms.untracked.bool(False),
    readBadChannels = cms.bool(False),
    CSCStripClusterChargeCut = cms.untracked.double(25.0)
)


