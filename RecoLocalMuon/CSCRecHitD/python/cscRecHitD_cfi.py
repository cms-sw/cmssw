import FWCore.ParameterSet.Config as cms

# parameters for CSC rechit building
csc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    CSCStripClusterSize = cms.untracked.int32(3),
    #
    #    Parameters for strip hits
    #
    CSCStripPeakThreshold = cms.untracked.double(10.0),
    #    constant systematics (in cm)
    ConstSyst = cms.untracked.double(0.03),
    readBadChannels = cms.bool(False),
    #
    #    Parameters for coordinate and uncertainty calculations
    #    Do not change them freely...
    #
    #    3 time bins noise (in ADC counts)      
    NoiseLevel = cms.untracked.double(7.0),
    CSCStripxtalksOffset = cms.untracked.double(0.03),
    #
    #    Parameters for 2-D hits (not used currently)
    #
    #  To be set once wire digis have timing info:
    CSCstripWireDeltaTime = cms.untracked.int32(8),
    #
    #    Calibration info:
    CSCUseCalibrations = cms.untracked.bool(True),
    #    a XT asymmetry model parameter 
    XTasymmetry = cms.untracked.double(0.005),
    #
    #    Which digis:
    #
    #  When using data from unpacker
    CSCStripDigiProducer = cms.string('muonCSCDigis'),
    CSCWireDigiProducer = cms.string('muonCSCDigis'),
    CSCDebug = cms.untracked.bool(False),
    #  This was not fully developed and currently NOT working
    CSCproduce1DHits = cms.untracked.bool(False),
    #
    #    Parameters for wire hits
    CSCWireClusterDeltaT = cms.untracked.int32(1),
    CSCStripClusterChargeCut = cms.untracked.double(25.0)
)


