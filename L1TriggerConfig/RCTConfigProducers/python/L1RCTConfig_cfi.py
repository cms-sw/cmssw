import FWCore.ParameterSet.Config as cms
from L1TriggerConfig.RCTConfigProducers.RCTCalibration_cff import *

RCTConfigProducers = cms.ESProducer("RCTConfigProducers",

    rct_calibration,

    eMaxForFGCut = cms.double(999.),
    eMinForFGCut = cms.double(6.),

    hOeCut = cms.double(0.05),
    eMaxForHoECut = cms.double(30.),
    eMinForHoECut = cms.double(1.),
    hMinForHoECut = cms.double(1.),

    eGammaLSB = cms.double(0.5),
    jetMETLSB = cms.double(0.5),

    jscQuietThresholdBarrel = cms.uint32(3),
    jscQuietThresholdEndcap = cms.uint32(3),
    hActivityCut = cms.double(4.0),
    eActivityCut = cms.double(4.0),
    eicIsolationThreshold = cms.uint32(7),

    noiseVetoHB = cms.bool(False),
    noiseVetoHEplus = cms.bool(False),
    noiseVetoHEminus = cms.bool(False),

    useCorrectionsLindsey = cms.bool(False),

    eGammaECalScaleFactors = cms.vdouble(1.09558, 1.08869, 1.09952, 1.10063,
                                         1.09825, 1.10759, 1.10656, 1.11377,
                                         1.11046, 1.11398, 1.12557, 1.14503,
                                         1.14956, 1.16809, 1.1787, 1.21008,
                                         1.19398, 1.24587, 1.30341, 1.4772,
                                         1.45471, 1.47039, 1.50559, 1.52439,
                                         1.57044, 1.58462, 1.62689, 1.62689),
    eGammaHCalScaleFactors = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0),
    jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0),
    jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0)
)



